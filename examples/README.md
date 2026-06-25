# ALLEX examples

Minimal, self-contained examples that load the published ALLEX model and replay a recorded
motion. No ROS, no sidecar, no internal packages — just the model files in this repository.

## `mujoco/replay.py` — recorded-motion playback (MuJoCo)

Dynamic playback: drives the recorded motion as the MJCF's position-actuator targets with
gravity-comp feed-forward (the same scheme the robot uses); `mj_step` enforces the passive
finger (`DIP←PIP`, `IP←MCP`) and waist couplings. Loads `mjcf/ALLEX.xml` (no ground plane —
ALLEX is fixed-base).

```bash
pip install mujoco numpy
python mujoco/replay.py --gui        # watch live (real-time); omit --gui to run headless
# options: --motion motions/<name>.npz   --model ../../mjcf/ALLEX.xml
```

Expect a few degrees of tracking lag during fast motion (PD control, not a kinematic teleport).

## `motions/`

Pre-sampled joint trajectories as `*.npz`:

| key | meaning |
|---|---|
| `joint_names` | `[ndof]` joint names — map the columns onto the MJCF/URDF/USD by name |
| `dt` | sample period [s] |
| `q` | `[frames, ndof]` joint positions [rad] |

These are dense **recorded** robot motions (sampled output), not generated here.

- `demo1.npz` — ~39 s ALLEX upper-body motion, 200 Hz.

---

PhysX (Isaac Sim) and Newton (MuJoCo-Warp) playback examples are on the way.
