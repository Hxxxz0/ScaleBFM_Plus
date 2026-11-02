"""Sample dense points on the collision surface of a packed library's objects (sidecar for the contact reward).

The surface_points of objects.json (256, visual mesh) are too sparse to measure finger-object distances on thin
parts (a 1.5 m tripod). Each collision mesh of an object's USD is simulated as its convex hull (PhysX convexHull
approximation, build_hdmi_objects.py): points are drawn uniformly by area on those hulls, in the object frame.

    <library>/object_contact_points.npy   (K, P, 3) float32, objects in the library's object_names order

    /isaac-sim/python.sh scripts/pretrain/data_process/add_contact_points.py --library <sm_lib>/hdmi_inspire_v1/train
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics
from scipy.spatial import ConvexHull


def hull_triangles(usd_path: str) -> np.ndarray:
    """(T, 3, 3) triangles of the convex hulls of the collision meshes, in the object frame."""
    stage = Usd.Stage.Open(usd_path)
    tris = []
    for prim in stage.Traverse():
        if not (prim.IsA(UsdGeom.Mesh) and prim.HasAPI(UsdPhysics.CollisionAPI)):
            continue
        points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=np.float64)
        xform = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()), dtype=np.float64)
        points = points @ xform[:3, :3] + xform[3, :3]  # row-vector convention of Gf matrices
        hull = ConvexHull(points)
        tris.append(points[hull.simplices])
    if not tris:
        raise ValueError(f"{usd_path}: no collision mesh")
    return np.concatenate(tris)


def sample_surface(tris: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    area = 0.5 * np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=-1)
    face = rng.choice(len(tris), size=n, p=area / area.sum())
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1
    u, v = np.where(flip, 1 - u, u), np.where(flip, 1 - v, v)
    t = tris[face]
    return t[:, 0] + u[:, None] * (t[:, 1] - t[:, 0]) + v[:, None] * (t[:, 2] - t[:, 0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", type=Path, required=True)
    ap.add_argument("--points", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    table = json.loads((args.library / "objects.json").read_text())["objects"]
    names = json.loads((args.library / "index.json").read_text()).get("object_names", list(table))
    rng = np.random.default_rng(args.seed)
    out = np.stack([sample_surface(hull_triangles(table[n]["usd_path"]), args.points, rng) for n in names]).astype(np.float32)
    np.save(args.library / "object_contact_points.npy", out)
    for n, pts in zip(names, out):
        print(f"{n:22s} extent {np.round(pts.max(0) - pts.min(0), 2)}")
    print(f"{out.shape} -> {args.library / 'object_contact_points.npy'}")


if __name__ == "__main__":
    main()
