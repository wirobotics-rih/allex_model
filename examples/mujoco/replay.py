"""Replay a baked ALLEX motion in MuJoCo — native position-actuator PD + gravity-comp playback.

Drives the recorded joint trajectory (motions/<name>.npz: joint_names + dt + q[T, ndof]
+ qd, optional kp, kv) with the MJCF's own `<position>` actuators (the native MuJoCo PD —
the same scheme the internal sweeps use, stable for the coupled finger/waist joints). Gravity
compensation (qfrc_bias) is fed forward on all dofs; `mj_step` enforces the passive finger
(`DIP←PIP`, `IP←MCP`) and waist equality couplings.

The D term is the **error derivative** kv·(q̇_ref − q̇): the position actuator already applies
kp·(target−q) − kv·q̇, so the reference velocity feed-forward +kv·q̇_ref (q̇_ref = the recorded
`qd`, the trajectory's analytic spline velocity) is added via qfrc_applied to recover the error
term — without it the velocity damping fights the reference and lags fast motion. PD gains `kp`/`kv` are taken **per
frame** from the recording when present (the motion changes them on the fly — e.g. low gains for
compliant phases) and written to the position actuator; otherwise the MJCF nominal gains. Torque
limits are the model's own (each joint's `actuatorfrcrange`), clamped by MuJoCo natively.

Loads `mjcf/ALLEX.xml` (the robot only, no ground plane) — ALLEX is fixed-base.

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
    qd = z["qd"].astype(np.float64)                        # reference vel (analytic spline velocity)
    kp_seq = z["kp"].astype(np.float64) if "kp" in z else None    # [T, ndof] per-frame PD gains (optional)
    kv_seq = z["kv"].astype(np.float64) if "kv" in z else None

    m = mujoco.MjModel.from_xml_path(args.model)
    m.opt.timestep = dt
    d = mujoco.MjData(m)
    sd = mujoco.MjData(m)                                   # scratch for gravity comp (qvel=0)

    # recorded (driven) joint -> (motion col, actuator id, dof addr, kv); mj_step resolves coupled joints.
    # Torque limits are the model's own (joint `actuatorfrcrange`, clamped by MuJoCo natively).
    pairs, qadr = [], {}
    for a in range(m.nu):
        jid = int(m.actuator_trnid[a, 0])
        jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
        qadr[jn] = int(m.jnt_qposadr[jid])
        if jn in names:
            kv = float(-m.actuator_biasprm[a, 2])          # position actuator: biasprm = [0, -kp, -kv]
            pairs.append((names.index(jn), a, int(m.jnt_dofadr[jid]), kv))

    def gravity_comp():
        sd.qpos[:] = d.qpos; sd.qvel[:] = 0.0
        mujoco.mj_forward(m, sd)
        return sd.qfrc_bias

    viewer = mujoco.viewer.launch_passive(m, d) if args.gui else None
    if viewer is not None:
        viewer.cam.distance = 3.0; viewer.cam.elevation = -20; viewer.cam.azimuth = 180
        viewer.cam.lookat[:] = [0.05, 0.0, 0.5]

    def step_to(fr, frd, kpf, kvf):
        d.qfrc_applied[:] = gravity_comp()
        for c, a, dof, kv0 in pairs:
            if kpf is not None:                            # per-frame gains -> set the position actuator
                kp = kpf[c]; kv = kvf[c]
                m.actuator_gainprm[a, 0] = kp; m.actuator_biasprm[a, 1] = -kp; m.actuator_biasprm[a, 2] = -kv
            else:
                kv = kv0                                   # MJCF nominal gains
            d.ctrl[a] = fr[c]                              # position target -> kp·(fr−q) − kv·q̇
            d.qfrc_applied[dof] += kv * frd[c]             # + kv·q̇_ref  ⇒  error-term D
        mujoco.mj_step(m, d)
        if viewer is not None:
            viewer.sync(); time.sleep(dt)                  # real-time pacing

    # ramp from the rest pose to the first recorded frame as PD targets — do NOT teleport qpos
    # (that would snap the equality-coupled joints and blow up the solver).
    home = np.array([d.qpos[qadr[n]] for n in names])
    nwarm = max(1, int(round(0.5 / dt)))
    zero = np.zeros(len(names))
    kp0 = kp_seq[0] if kp_seq is not None else None
    kv0 = kv_seq[0] if kv_seq is not None else None
    for k in range(nwarm):
        a = (k + 1) / nwarm
        step_to(home * (1.0 - a) + q[0] * a, zero, kp0, kv0)   # slow ramp: no velocity feed-forward
    for i in range(len(q)):
        step_to(q[i], qd[i],
                kp_seq[i] if kp_seq is not None else None,
                kv_seq[i] if kv_seq is not None else None)
    print(f"[replay] played {len(q)} frames ({len(q) * dt:.2f}s) of {Path(args.motion).name}", flush=True)
    if viewer is not None:
        viewer.close()


if __name__ == "__main__":
    main()
