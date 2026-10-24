"""FK of the MJCF tree against the IsaacLab FK replay stored in the motion npz files."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
from sm_offline import ASSET_XML, load  # noqa: E402

# A retargeted G1 motion npz with the IsaacLab FK replay, e.g. lafan_g1_v1/motions/dance2_subject3.npz.
MOTION = os.environ.get("SCALETRACK_TEST_MOTION")


@pytest.mark.skipif(not MOTION, reason="set SCALETRACK_TEST_MOTION to a motion npz")
def test_fk_matches_isaaclab_replay():
    kin = load("kinematics")
    rot = load("rotations")
    z = np.load(MOTION)
    bodies = [str(x) for x in z["robot_body_names"]]
    joints = [str(x) for x in z["robot_joint_names"]]
    sl = slice(100, 2100)
    pos = torch.tensor(z["body_pos_w"][sl], dtype=torch.float64)
    quat = torch.tensor(z["body_quat_w"][sl], dtype=torch.float64)
    ang = torch.tensor(z["body_ang_vel_w"][sl], dtype=torch.float64)
    jp = torch.tensor(z["joint_pos"][sl], dtype=torch.float64)
    jv = torch.tensor(z["joint_vel"][sl], dtype=torch.float64)
    fk = kin.HumanoidKinematics(str(ASSET_XML), joints, bodies, "cpu")
    for k in ("local_pos", "local_quat", "axes"):
        setattr(fk, k, getattr(fk, k).double())
    w_b = rot.quat_apply_inverse(quat[:, 0], ang[:, 0])
    out = fk.forward(pos[:, 0], quat[:, 0], jp, torch.zeros_like(pos[:, 0]), w_b, jv)
    pos_err = (out["pos"] - pos).norm(dim=-1).max().item()
    rot_err = rot.quat_angle_sq(out["quat"], quat).sqrt().max().item()
    ang_err = (out["ang_vel"] - ang).norm(dim=-1).max().item()
    print(f"max body pos err {pos_err:.2e} m, rot err {rot_err:.2e} rad, ang vel err {ang_err:.2e} rad/s")
    assert pos_err < 1e-4 and rot_err < 1e-4 and ang_err < 1e-3


if __name__ == "__main__":
    test_fk_matches_isaaclab_replay()
