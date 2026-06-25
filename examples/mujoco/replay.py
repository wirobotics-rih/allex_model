"""Replay a baked ALLEX motion in MuJoCo — dynamic PD + gravity-comp playback.

Drives the recorded joint trajectory (motions/<name>.npz: joint_names + dt + q[T, ndof], rad)
as the MJCF's position-actuator targets with gravity compensation (qfrc_bias) fed forward;
mj_step enforces the passive finger (`DIP←PIP`, `IP←MCP`) and waist equality couplings. Same
scheme the ALLEX robot and the internal sim sweeps use.

Loads `mjcf/ALLEX.xml` (the robot only, no ground plane) — ALLEX is fixed-base, so none is
needed and a floor at the base would just blow up the contacts.

Run:  python replay.py [--motion ../motions/demo1.npz] [--model ../../mjcf/ALLEX.xml] [--gui]
Requires only `pip install mujoco numpy`.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(HERE.parents[1] / "mjcf" / "ALLEX.xml"))
    ap.add_argument("--motion", default=str(HERE.parent / "motions" / "demo1.npz"))
    ap.add_argument("--gui", action="store_true", help="watch live (real-time) in the MuJoCo viewer")
    args = ap.parse_args()

    z = np.load(args.motion, allow_pickle=True)
    names = [str(x) for x in z["joint_names"]]
    q = z["q"].astype(np.float64)                          # [T, ndof] rad
    dt = float(z["dt"])

    m = mujoco.MjModel.from_xml_path(args.model)
    m.opt.timestep = dt
    d = mujoco.MjData(m)
    sd = mujoco.MjData(m)                                   # scratch for gravity comp (qvel=0)

    # recorded (driven) joint -> its position actuator; mj_step resolves the passive coupled joints
    pairs, qadr = [], {}
    for a in range(m.nu):
        jid = int(m.actuator_trnid[a, 0])
        jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
        qadr[jn] = int(m.jnt_qposadr[jid])
        if jn in names:
            pairs.append((names.index(jn), a))

    def gravity_comp():
        sd.qpos[:] = d.qpos; sd.qvel[:] = 0.0
        mujoco.mj_forward(m, sd)
        return sd.qfrc_bias

    viewer = mujoco.viewer.launch_passive(m, d) if args.gui else None
    if viewer is not None:
        viewer.cam.distance = 3.0; viewer.cam.elevation = -20; viewer.cam.azimuth = 120
        viewer.cam.lookat[:] = [0.0, 0.0, 0.9]

    def step_to(fr):
        for c, a in pairs:
            d.ctrl[a] = fr[c]
        d.qfrc_applied[:] = gravity_comp()
        mujoco.mj_step(m, d)
        if viewer is not None:
            viewer.sync(); time.sleep(dt)                  # real-time pacing

    # ramp from the rest pose to the first recorded frame as PD targets — do NOT teleport qpos
    # (that would snap the equality-coupled joints and blow up the solver).
    home = np.array([d.qpos[qadr[n]] for n in names])
    nwarm = max(1, int(round(0.5 / dt)))
    for k in range(nwarm):
        a = (k + 1) / nwarm
        step_to(home * (1.0 - a) + q[0] * a)
    for i in range(len(q)):
        step_to(q[i])
    print(f"[replay] played {len(q)} frames ({len(q) * dt:.2f}s) of {Path(args.motion).name}", flush=True)
    if viewer is not None:
        viewer.close()


if __name__ == "__main__":
    main()
