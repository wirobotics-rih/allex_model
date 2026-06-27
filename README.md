# ALLEX Simulation Model

[![License](https://img.shields.io/github/license/wirobotics-rih/allex_model)](./LICENSE)

**ALLEX** is an upper-body humanoid robot by WIRobotics, built for dexterous
manipulation and learning-based robotics. It combines whole-body compliance across the
arms, hands, neck, and waist with human-level dexterity from two high-DOF hands, and is
engineered for high backdrivability and low distal mass to keep the sim-to-real gap small.

This repository is a multi-format simulation model of ALLEX, provided in **MJCF**, **URDF**,
and **USD** with a shared mesh set. The same kinematics, inertials, and joint coupling are
projected into each format so the robot behaves consistently across MuJoCo and the
Isaac / OpenUSD ecosystem, and stays consistent for ROS 2 tooling (RViz, MoveIt).

- **48 actuated DOF** — two 7-DOF arms, two 15-DOF five-finger hands, a 2-DOF waist, and a
  2-DOF neck.
- 60 revolute joints total; 12 are passively coupled — 10 finger couplings plus
  2 waist-linkage joints (see [Joint coupling](#joint-coupling)).
- Current version: **v0.1.3** (pre-release; see [Versioning](#versioning)).

> This is a simulation model of **ALLEX**, the humanoid robot under development at
> WIRobotics. The hardware is currently at **research-prototype v2**, with a limited
> **research edition** shipping to research partners later this year. These files are a
> research/simulation model provided for evaluation and to demonstrate a small sim-to-real
> gap — not the production/manufacturing model.

## Layout

```
allex_model/
├─ mjcf/
│  ├─ ALLEX.xml      # robot model (MuJoCo)
│  └─ scene.xml      # ALLEX.xml + ground plane, lighting (load this one)
├─ urdf/
│  └─ ALLEX.urdf     # robot model (ROS 2 / RViz / MoveIt — kinematics, not dynamics)
├─ usd/
│  └─ ALLEX.usd      # robot model (Isaac Sim / OpenUSD) — EXPERIMENTAL
├─ meshes/           # tessellated visual + collision geometry (STL)
└─ LICENSE
```

## Quick start

**MuJoCo**
```python
import mujoco
model = mujoco.MjModel.from_xml_path("mjcf/scene.xml")
data  = mujoco.MjData(model)
# or: python -m mujoco.viewer --mjcf=mjcf/scene.xml
```

**ROS 2 / RViz / MoveIt**

This repository ships the model files only — it is **not** a ROS package (no `package.xml`).
The URDF references meshes as `package://allex_description/...`, so loading it in RViz
requires the `allex_description` package on your ament path. That package — together with
launch files, `ros2_control`, gravity compensation, and MoveIt configuration — is provided
by the separate `allex_ros2` workspace (published separately), which consumes this model.

The URDF is intended for **kinematics and integration** — RViz visualization, MoveIt
planning, and ROS 2 — not dynamics. **Gazebo is not supported** (its physics does not
reliably handle the coupling joints; see [Per-format notes](#per-format-notes)). For
dynamics, use MJCF or USD.

**Isaac Sim / OpenUSD** *(experimental)*
```python
from pxr import Usd
stage = Usd.Stage.Open("usd/ALLEX.usd")
```
> The USD export is experimental and still being validated — see
> [Per-format notes](#per-format-notes). Prefer MJCF or URDF for now.

## Per-format notes

All formats share the same base conventions — **meters / kilograms**, **Z** up, root link
**`Base_Link`** — and the same kinematics and inertials. Control-related fields and units,
however, differ by runtime:

- **MJCF (MuJoCo).** Position actuators carry PD gains (`kp`/`kv`, radian-space), torque
  limits (`actuatorfrcrange`), and joint ranges. MuJoCo has no joint *velocity*-limit
  field, so velocity limits are **not** encoded here — enforce them in your controller if
  needed.
- **URDF (ROS 2 — RViz / MoveIt).** Per-joint position, effort, and velocity limits are
  included; PD gains are **not** in the URDF (they live in the `ros2_control` config of the
  separate `allex_ros2` workspace). Meshes are referenced via `package://allex_description`
  (see [Quick start](#quick-start)). Use the URDF for **kinematics and integration**
  (visualization, motion planning, ROS 2) — **not dynamics**. **Gazebo is not supported:**
  its physics (gz-physics on Jazzy) does not reliably handle the mimic/coupling joints —
  the finger DIP/IP couplings and the waist 4-bar linkage — so contacts, and especially
  gravity compensation propagated through the waist coupling, are wrong. Use **MJCF** or
  **USD** for dynamics.
- **USD (Isaac Sim / OpenUSD) — experimental.** Both a PhysX and a Newton/MJC schema are
  emitted. Drive gains follow the UsdPhysics convention — **per-degree** units (the
  radian-space `kp`/`kv` scaled by π/180) — and the same values apply to both the PhysX and
  Newton backends, which convert to their internal units on import. The finger coupling is a
  linear approximation in both schemas, matching the URDF (see
  [Joint coupling](#joint-coupling)). Newton support in Isaac Sim is itself experimental —
  prefer MJCF or URDF for now.

## Joint coupling

Two passive couplings are baked into the model:

- **Fingers — 10 joints.** Each finger's distal joint follows its proximal one
  (finger `DIP ← PIP`, thumb `IP ← MCP`).
- **Waist — 2 joints.** The upper and dummy pitch joints follow `Waist_Lower_Pitch` at a
  ±1 ratio (a parallel pitch linkage). This relation is linear and exact in every format.

The finger coupling is the same physical relationship everywhere, but it is projected
differently to match each runtime:

| Format | Finger coupling representation |
|---|---|
| MJCF | exact **polynomial** (`equality` joint, quartic) |
| URDF, USD (PhysX and Newton schemas) | **linear approximation** (least-squares fit through the origin over the full range of motion) |

Only the MJCF carries the exact polynomial; for URDF and USD consumers the finger coupling
is a linear approximation of it. Expect a small deviation away from the origin — if
your application is sensitive to fingertip kinematics, validate against the MJCF model.

## Versioning

Releases are tagged `vMAJOR.MINOR.PATCH`. The model is **pre-release** while ALLEX is in
development: `0.MINOR` tracks the hardware prototype generation (`0.1` = research-prototype
v1, `0.2` = research-prototype v2), each maintained on its own branch (`proto_v1`,
`proto_v2`); `main` follows the latest generation. `1.0.0` will mark the research-edition
release.

## Contributing

These files are generated from an internal single source of truth and published here as a
read-only mirror, so please **do not edit the model files directly** — changes belong
upstream and would otherwise be overwritten on the next release. For corrections, bug
reports, or feature requests (missing joint limits, coupling accuracy, added formats, etc.),
open an issue in this repository or contact the WIRobotics simulation team.

## License

The robot description and configuration files (`mjcf/`, `urdf/`, `usd/`) are released under
the **BSD 3-Clause License** — see [LICENSE](./LICENSE). The 3D mesh geometry
(`meshes/*.stl`, both visual and collision) is **not** BSD-licensed; it is provided as
reference geometry for evaluation only, under separate terms — see
[MESHES-LICENSE](./MESHES-LICENSE).
