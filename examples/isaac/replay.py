"""Replay a baked ALLEX motion in Isaac Sim (PhysX): gravity compensation + native PD.

Drives the recorded joint trajectory (motions/<name>.npz: joint_names + dt + q[T, ndof]
+ qd) with the USD's **native PhysX position drive** (the model's own kp/kv + maxForce
baked in by the emitter — no override), **gravity ON**, and gravity compensation fed forward as
joint efforts (MuJoCo qfrc_bias). The model's PhysX mimic couplings resolve the finger
(`DIP←PIP`, `IP←MCP`) / waist joints automatically.

Per step: gravity-comp effort + PD position target + reference-velocity target (q̇_ref), then
`world.step`. The D term is the **error derivative** kv·(q̇_ref − q̇): the drive applies
damping·(q̇_target − q̇), so feeding q̇_ref as the velocity target recovers it (matching the MuJoCo
example). The gains and torque limits are the model's own — this example does not change them.

Run:  python replay.py [--motion ../motions/hello.npz] [--gui]
        [--usd ../../usd/ALLEX.usd] [--mjcf ../../mjcf/ALLEX.xml]
Requires an Isaac Sim Python env with `mujoco` installed (USD = PhysX dynamics; MJCF = the
gravity-comp model). No ROS, no sidecar, no internal packages.
"""
import argparse
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PRIM = "/World/ALLEX"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--usd", default=str(HERE.parents[1] / "usd" / "ALLEX.usd"))
    ap.add_argument("--mjcf", default=str(HERE.parents[1] / "mjcf" / "ALLEX.xml"))
    ap.add_argument("--motion", default=str(HERE.parent / "motions" / "hello.npz"))
    ap.add_argument("--gui", action="store_true", help="watch live in the Isaac Sim viewport")
    args = ap.parse_args()

    from isaacsim import SimulationApp
    sim = SimulationApp({"headless": not args.gui})        # MUST precede any isaacsim.* import
    from isaacsim.core.api import World
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.core.prims import Articulation
    import mujoco

    z = np.load(args.motion, allow_pickle=True)
    names = [str(x) for x in z["joint_names"]]
    q = z["q"].astype(np.float32)
    dt = float(z["dt"])
    qd = z["qd"].astype(np.float32)                        # reference vel (analytic spline velocity)

    m = mujoco.MjModel.from_xml_path(args.mjcf)             # gravity-comp oracle (qfrc_bias at qvel=0)
    md = mujoco.MjData(m)

    add_reference_to_stage(args.usd, PRIM)
    world = World(physics_dt=dt, rendering_dt=dt, stage_units_in_meters=1.0)  # gravity ON
    art = Articulation(prim_paths_expr=PRIM, name="allex"); world.scene.add(art)
    world.reset()

    if args.gui:                                           # USD ships no light/camera -> add for viewport
        from pxr import UsdLux, Sdf
        from isaacsim.core.utils.stage import get_current_stage
        from isaacsim.core.utils.viewports import set_camera_view
        stage = get_current_stage()
        UsdLux.DistantLight.Define(stage, Sdf.Path("/World/Key")).CreateIntensityAttr(2500.0)
        UsdLux.DomeLight.Define(stage, Sdf.Path("/World/Sky")).CreateIntensityAttr(300.0)
        set_camera_view(eye=[2.78, 0.0, 1.49], target=[0.05, 0.0, 0.5])

    dof = list(art.dof_names); ndof = len(dof)
    qadr = np.full(ndof, -1, int); dadr = np.full(ndof, -1, int)   # PhysX dof -> MuJoCo qpos/dof addr
    for i, jn in enumerate(dof):
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if j >= 0:
            qadr[i] = int(m.jnt_qposadr[j]); dadr[i] = int(m.jnt_dofadr[j])
    pairs = [(c, dof.index(n)) for c, n in enumerate(names) if n in dof]   # motion col -> PhysX dof
    # Gains and torque limits are the model's own (the USD drive kp/kv + maxForce baked in by the
    # emitter) — this example uses them as-is and does not call set_gains.

    def gravity_ff(qP):                                    # qP: measured positions (PhysX dof order)
        md.qpos[:] = 0.0; md.qvel[:] = 0.0
        for i in range(ndof):
            if qadr[i] >= 0: md.qpos[qadr[i]] = qP[i]
        mujoco.mj_forward(m, md)
        g = np.zeros((1, ndof), np.float32)
        for i in range(ndof):
            if dadr[i] >= 0: g[0, i] = md.qfrc_bias[dadr[i]]
        return g

    def targets(fr):
        t = np.zeros((1, ndof), np.float32)
        for c, di in pairs:
            t[0, di] = fr[c]
        return t

    def step_to(fr, frd):
        cur = art.get_joint_positions()
        if cur is not None:
            art.set_joint_efforts(gravity_ff(cur[0]))      # gravity comp feed-forward
        art.set_joint_position_targets(targets(fr))        # PD position target (native USD drive)
        art.set_joint_velocity_targets(targets(frd))       # q̇_ref velocity target -> error-term D
        world.step(render=args.gui)

    # ramp from the default pose to the first recorded frame as PD targets — do NOT teleport
    # joint positions, which would snap the passive mimic-coupled joints and blow up the solver.
    cur = art.get_joint_positions()
    home = (np.array([cur[0][dof.index(n)] for n in names], np.float32)
            if cur is not None else np.zeros(len(names), np.float32))
    nwarm = max(1, int(round(0.5 / dt)))
    zero = np.zeros(len(names), np.float32)
    for k in range(nwarm):
        a = (k + 1) / nwarm
        step_to(home * (1.0 - a) + q[0] * a, zero)
    for i in range(len(q)):
        step_to(q[i], qd[i])
    print(f"[replay] played {len(q)} frames ({len(q) * dt:.2f}s) of {Path(args.motion).name}", flush=True)
    sim.close()


if __name__ == "__main__":
    main()
