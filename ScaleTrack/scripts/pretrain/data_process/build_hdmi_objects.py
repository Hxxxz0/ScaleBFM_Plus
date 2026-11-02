"""Build the rigid-object assets of HDMI's any4hdmi HOI datasets (OMOMO rigid5, Weave reference) for IsaacLab.

For every dataset the robot + object MJCF is compiled with MuJoCo, and the free ``object`` body is exported with its
exact physics: mass, CoM, principal inertia and friction from the compiled model, collision geometry from its
collision geoms (meshes or primitives, each an IsaacLab convex hull), visual mesh for surface sampling. Writes

    <output_dir>/<name>.usda       geometry only (collision meshes + visual mesh); rigid body / mass / material are
                                   applied at spawn time from objects.json (supermimic.hoi_env.spawn_dynamic_object)
    <output_dir>/objects.json      per object: usd_path, dynamics, dimensions_m, hull_points (convex hull of the
                                   collision vertices, object frame), surface_points (256, seed 0), shape_bps (32)

Run with the Isaac Sim python (mujoco, trimesh, scipy, pxr):

    /isaac-sim/python.sh scripts/pretrain/data_process/build_hdmi_objects.py \
        --datasets <data_root>/any4hdmi --output_dir <data_root>/any4hdmi/objects
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import trimesh
from pxr import Gf, Usd, UsdGeom, UsdPhysics, Vt
from scipy.spatial import ConvexHull

NUM_SURFACE_POINTS = 256
NUM_BPS = 32
BPS_HALF_EXTENT = 1.0  # m, fixed basis cube shared by all objects (encodes size and shape)


def quat_to_mat(q):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, dtype=np.float64))
    return m.reshape(3, 3)


def geom_mesh(model, g):
    """Vertices (body frame) and faces of geom g."""
    t = model.geom_type[g]
    R, p = quat_to_mat(model.geom_quat[g]), model.geom_pos[g]
    if t == mujoco.mjtGeom.mjGEOM_MESH:
        m = model.geom_dataid[g]
        v = model.mesh_vert[model.mesh_vertadr[m] : model.mesh_vertadr[m] + model.mesh_vertnum[m]]
        f = model.mesh_face[model.mesh_faceadr[m] : model.mesh_faceadr[m] + model.mesh_facenum[m]]
        return v @ R.T + p, f
    size = model.geom_size[g]
    if t == mujoco.mjtGeom.mjGEOM_BOX:
        mesh = trimesh.creation.box(extents=2 * size)
    elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=size[0])
    elif t == mujoco.mjtGeom.mjGEOM_CYLINDER:
        mesh = trimesh.creation.cylinder(radius=size[0], height=2 * size[1], sections=24)
    elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
        mesh = trimesh.creation.capsule(radius=size[0], height=2 * size[1], count=[12, 12])
    else:
        raise ValueError(f"unsupported geom type {t}")
    return np.asarray(mesh.vertices) @ R.T + p, np.asarray(mesh.faces)


def object_from_mjcf(xml: Path):
    model = mujoco.MjModel.from_xml_path(str(xml))
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    if body < 0:
        raise ValueError(f"{xml}: no 'object' body")
    geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == body]
    collision = [g for g in geoms if model.geom_contype[g] or model.geom_conaffinity[g]]
    visual = [g for g in geoms if g not in collision and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
    if not collision:
        raise ValueError(f"{xml}: object has no collision geom")
    friction = float(model.geom_friction[collision[0]][0])
    dynamics = dict(
        mass_kg=float(model.body_mass[body]),
        com_local_m=model.body_ipos[body].tolist(),
        diagonal_inertia_kg_m2=model.body_inertia[body].tolist(),
        principal_axes_wxyz=model.body_iquat[body].tolist(),
        static_friction=friction,
        dynamic_friction=friction,
        restitution=0.0,
        source=str(xml),
    )
    return dict(
        collision=[geom_mesh(model, g) for g in collision],
        visual=[geom_mesh(model, g) for g in (visual or collision)],
        dynamics=dynamics,
    )


def write_usd(path: Path, collision, visual):
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Object")
    stage.SetDefaultPrim(root.GetPrim())

    def mesh(prim_path, v, f):
        m = UsdGeom.Mesh.Define(stage, prim_path)
        m.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, x)) for x in v]))
        m.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(f)))
        m.CreateFaceVertexIndicesAttr(Vt.IntArray([int(i) for i in np.asarray(f).reshape(-1)]))
        m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        return m

    for i, (v, f) in enumerate(collision):
        m = mesh(f"/Object/collision_{i}", v, f)
        UsdPhysics.CollisionAPI.Apply(m.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr(UsdPhysics.Tokens.convexHull)
        UsdGeom.Imageable(m).MakeInvisible()
    for i, (v, f) in enumerate(visual):
        mesh(f"/Object/visual_{i}", v, f)
    stage.GetRootLayer().Save()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, required=True)
    a = ap.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    bps_basis = rng.uniform(-BPS_HALF_EXTENT, BPS_HALF_EXTENT, size=(NUM_BPS, 3))
    entries = {}
    sources = []
    for d in sorted(a.datasets.glob("omomo_*")):
        sources.append((f"omomo_{d.name.split('_', 1)[1]}", d))
    weave = a.datasets / "weave_reference_1605"
    for d in sorted(p for p in weave.iterdir() if (p / "manifest.json").exists()):
        sources.append((f"weave_{d.name}", d))
    for name, d in sources:
        manifest = json.loads((d / "manifest.json").read_text())
        obj = object_from_mjcf(d / manifest["mjcf"])
        usd = a.output_dir / f"{name}.usda"
        write_usd(usd, obj["collision"], obj["visual"])
        coll = np.concatenate([v for v, _ in obj["collision"]])
        hull = coll[ConvexHull(coll).vertices]
        vis = trimesh.util.concatenate([trimesh.Trimesh(v, f, process=False) for v, f in obj["visual"]])
        surface, _ = trimesh.sample.sample_surface(vis, NUM_SURFACE_POINTS, seed=0)
        dense, _ = trimesh.sample.sample_surface(vis, 4096, seed=1)
        bps = np.linalg.norm(bps_basis[:, None] - dense[None], axis=-1).min(-1)
        entries[name] = dict(
            usd_path=str(usd),
            dataset_dir=str(d),
            dynamics=obj["dynamics"],
            dimensions_m=(hull.max(0) - hull.min(0)).tolist(),
            hull_points=hull.tolist(),
            surface_points=np.asarray(surface).tolist(),
            shape_bps=bps.tolist(),
            num_collision_meshes=len(obj["collision"]),
        )
        print(f"{name}: mass={obj['dynamics']['mass_kg']:.3f} collision={len(obj['collision'])} "
              f"dims={np.round(entries[name]['dimensions_m'], 3).tolist()} hull={len(hull)}", flush=True)
    meta = dict(objects=entries, bps_basis=bps_basis.tolist(), bps_half_extent=BPS_HALF_EXTENT)
    (a.output_dir / "objects.json").write_text(json.dumps(meta))
    print(f"wrote {len(entries)} objects to {a.output_dir}", flush=True)


if __name__ == "__main__":
    main()
