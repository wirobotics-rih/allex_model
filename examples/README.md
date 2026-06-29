# ALLEX examples

Minimal, self-contained examples that load the published ALLEX model and replay a recorded
motion. No ROS, no sidecar, no internal packages — just the model files in this repository.

Each backend uses its own Python environment (they are not interchangeable — e.g. Isaac Sim
pins Python 3.12). Install the simulator per its official guide; see **Requirements** in each
section below. The MuJoCo example is the simplest and the recommended starting point.

## `mujoco/replay.py` — recorded-motion playback (MuJoCo)

Dynamic playback with native MuJoCo actuators: each joint is a position servo (kp) **plus a
velocity servo (kv)**, so the damping is the error term `kv·(q̇_ref − q̇)` integrated *implicitly*
(stable at any speed) — not an explicit feed-forward (which diverges on light/fast joints). Gravity
comp is fed forward, and each step the joint's `actuatorfrcrange` is shifted to `[-τ-G, +τ-G]` so the
single motor limit clamps gravity-comp + PD **together** (`clip(PD + G, ±τ)`). `mj_step` enforces
the passive finger (`DIP←PIP`, `IP←MCP`) and waist couplings. Per-frame `kp`/`kv` from the motion are
applied when present. The position(kp,kv) → position(kp)+velocity(kv) split is built at load via
`MjSpec` (the published MJCF is unchanged). Loads `mjcf/ALLEX.xml` (no ground plane — ALLEX is
fixed-base).

**Requirements:** Python ≥ 3.10; `pip install mujoco numpy` (see the
[MuJoCo install docs](https://mujoco.readthedocs.io/en/stable/python.html)). No GPU needed.
Tested with MuJoCo 3.5.0 and NumPy 2.3.1 on Python 3.12.

```bash
python mujoco/replay.py --gui        # watch live (real-time); omit --gui to run headless
# options: --motion motions/<name>.npz   --model ../../mjcf/ALLEX.xml
```

Expect a few degrees of tracking lag during fast motion (PD control, not a kinematic teleport).

## `isaac/replay.py` — recorded-motion playback (Isaac Sim / PhysX)

Loads the USD and replays the trajectory under PhysX with **gravity on**: the USD's native
position drive does the PD (torque capped by the drive's `maxForce` — the model's own torque
limit), with gravity comp + error-term D fed forward as joint efforts / velocity targets — the
same scheme the robot uses — so the arms hold against gravity while tracking. This example uses the
model's **nominal gains** (it does not apply the motion's per-frame `kp`/`kv`), which is why it ships
with `hello.npz`. The drive's `maxForce` clamps the **PD only** (gravity is a separate effort on top);
PhysX's drive limit is symmetric, so it can't do the combined `clip(PD + G, ±τ)` headroom the MuJoCo
example does — an accepted difference, as PhysX is not the accuracy reference. The model's PhysX mimic
couplings resolve the finger/waist joints automatically. Expect small tracking lag during fast motion.

**Requirements:** NVIDIA Isaac Sim 6.x on Python 3.12 — install per the
[Isaac Sim install docs](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_python.html);
also `pip install mujoco` (used for the gravity-comp model). Needs an NVIDIA GPU. Developed
against Isaac Sim 6.x — still being validated.

```bash
python isaac/replay.py --gui          # in an Isaac Sim Python env; omit --gui for headless
```

The gravity-comp model is the repo's `mjcf/ALLEX.xml` (its masses match the USD). The USD
export is experimental — prefer the MuJoCo example for now.

## `motions/`

ALLEX supports **MIT Mode** actuator control — a feed-forward torque plus a PD term,
`τ = τ_ff + kp·(q_des − q) + kv·(q̇_des − q̇)`. These demos are a simple showcase of it: each
example applies a gravity-compensation torque as the feed-forward `τ_ff` on top of PD tracking of
the recorded position / velocity targets (`q_des`/`q̇_des`) — the same control scheme the robot
runs. The motions below are pre-sampled joint trajectories as `*.npz`:

| key | meaning |
|---|---|
| `joint_names` | `[ndof]` joint names — map the columns onto the MJCF/URDF/USD by name |
| `dt` | sample period [s] |
| `q` | `[frames, ndof]` joint positions [rad] |
| `qd` | `[frames, ndof]` reference joint velocity [rad/s] (error-term D reference) |
| `kp` | `[frames, ndof]` per-frame PD position gain (optional; the motion lowers it for compliant phases) |
| `kv` | `[frames, ndof]` per-frame PD velocity gain (optional) |

Dense joint trajectories sampled at 200 Hz. `qd` is the **analytic spline velocity** from the bake
(the trajectory's own derivative, not a finite difference); the examples feed it as the error-term
D reference. `kp`/`kv` appear only when a motion drives the gains on the fly; without them the
examples use the model's nominal gains.

- `demo1.npz` — ~43 s ALLEX upper-body motion with per-frame `kp`/`kv` (the motion lowers the gains
  for compliant phases). Used by the **MuJoCo** example.
- `hello.npz` — ~8 s right-hand wave at the model's **nominal gains**. Used by the **Isaac / PhysX**
  example: PhysX's implicit articulation drive tracks stiff nominal-gain motions well, but lags the
  low-gain compliant phases of `demo1` (which the MuJoCo example handles fine).
