"""Replay a baked ALLEX motion in MuJoCo — position+velocity actuator PD + gravity-comp playback.

Drives the recorded joint trajectory (motions/<name>.npz: joint_names + dt + q[T, ndof] + qd,
optional kp, kv) with native MuJoCo actuators. Each joint's PD is a position servo (kp) plus a
velocity servo (kv): position ctrl = recorded q, velocity ctrl = recorded q̇, so the damping is
the error term kv·(q̇_ref − q̇), integrated IMPLICITLY by MuJoCo (stable at any speed). An explicit
qfrc feed-forward +kv·q̇_ref would instead diverge on light/fast joints once dt > 2·I/kv; the
implicit servo never does. `mj_step` enforces the passive finger (DIP←PIP, IP←MCP) and waist
equality couplings.

Gravity compensation (qfrc_bias) is fed forward on all dofs, and the single motor torque limit
clamps gravity-comp + PD together: each step the joint's actuatorfrcrange is shifted to
[-τ-G, +τ-G] so clip(PD, that) + G = clip(PD + G, ±τ). (τ already includes the gravcomp budget for
the electrically-gravity-compensated joints, so the same shift applies uniformly.) PD gains kp/kv
are taken per frame from the recording when present (the motion changes them on the fly — e.g. low
gains for compliant phases), else the MJCF nominal gains.

The position(kp,kv) → position(kp)+velocity(kv) split is built at load time via MjSpec; the
published MJCF is unchanged. Loads mjcf/ALLEX.xml (robot only, no ground plane) — ALLEX is fixed-base.

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


def build_model(path):
    """Load the MJCF and split each position(kp,kv) actuator into position(kp) + velocity(kv) so
    kv·(q̇_ref − q̇) is an implicit velocity servo (stable). Driver-local; the MJCF is unchanged."""
    spec = mujoco.MjSpec.from_file(str(path))
    split = []
    for a in spec.actuators:
        split.append((a.target, -a.biasprm[2]))            # (joint, kv); position bias = [0, -kp, -kv]
        bp = list(a.biasprm); bp[2] = 0.0; a.biasprm = bp  # position servo: kp only now
    for jn, kv in split:                                   # add a velocity servo per joint
        va = spec.add_actuator()
        va.name = jn.replace("_Joint", "") + "_Velocity"
        va.trntype = mujoco.mjtTrn.mjTRN_JOINT; va.target = jn
        va.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        gp = [0.0] * len(va.gainprm); gp[0] = kv; va.gainprm = gp
        va.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        bp = [0.0] * len(va.biasprm); bp[2] = -kv; va.biasprm = bp   # force = kv·(ctrl − q̇)
    return spec.compile()


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

    m = build_model(args.model)
    m.opt.timestep = dt
    d = mujoco.MjData(m)
    sd = mujoco.MjData(m)                                   # scratch for gravity comp (qvel=0)

    # recorded joint -> position + velocity actuator ids; mj_step resolves the coupled joints.
    pos_act, vel_act, qadr = {}, {}, {}
    for a in range(m.nu):
        jid = int(m.actuator_trnid[a, 0])
        jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
        qadr[jn] = int(m.jnt_qposadr[jid])
        if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a).endswith("_Velocity"):
            vel_act[jn] = a
        else:
            pos_act[jn] = (a, jid)
    pairs = []   # (motion col, position actuator, velocity actuator, dof addr, joint id, τ or None)
    for jn, (pa, jid) in pos_act.items():
        if jn in names:
            tau = float(m.jnt_actfrcrange[jid, 1]) if m.jnt_actfrclimited[jid] else None
            pairs.append((names.index(jn), pa, vel_act[jn], int(m.jnt_dofadr[jid]), jid, tau))

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
        for c, pa, va, dof, jid, tau in pairs:
            if kpf is not None:                            # per-frame dynamic gains -> set both servos
                kp, kv = kpf[c], kvf[c]
                m.actuator_gainprm[pa, 0] = kp; m.actuator_biasprm[pa, 1] = -kp   # position: kp·(ctrl−q)
                m.actuator_gainprm[va, 0] = kv; m.actuator_biasprm[va, 2] = -kv   # velocity: kv·(ctrl−q̇)
            d.ctrl[pa] = fr[c]                             # position target q_des
            d.ctrl[va] = frd[c]                            # velocity target q̇_ref -> error-term D
            if tau is not None:                            # headroom: clip(PD, [-τ-G, τ-G]) + G = clip(PD+G, ±τ)
                g = d.qfrc_applied[dof]
                m.jnt_actfrcrange[jid] = (-tau - g, tau - g)
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
        step_to(home * (1.0 - a) + q[0] * a, zero, kp0, kv0)   # slow ramp: zero velocity target
    for i in range(len(q)):
        step_to(q[i], qd[i],
                kp_seq[i] if kp_seq is not None else None,
                kv_seq[i] if kv_seq is not None else None)
    print(f"[replay] played {len(q)} frames ({len(q) * dt:.2f}s) of {Path(args.motion).name}", flush=True)
    if viewer is not None:
        viewer.close()


if __name__ == "__main__":
    main()
