# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

- **Python env**: conda `lerobot_py310` (Python 3.10). All scripts must run inside it.
- **ROS 2 distro**: source ROS 2 before running any ROS-dependent script (observer, deploy, subscribe_and_viz).
- **Network**: robot at `ws://10.192.1.2:5000`. Disable ufw and unset proxy vars before connecting (`sudo ufw disable`; `export NO_PROXY=10.192.1.2,...`). `WSController.start()` scrubs proxy env vars automatically, but `tron2_ws.run()` does not — run `test.sh` first if using that path.

## Run order (once per power-on session)

```bash
python start.py      # walk arm [0]*14 → WP3 (~26 s), then exits
python deploy.py     # GR00T inference loop until Ctrl+C
python shutdown.py   # reverse WP3 → [0]*14, then exits
```

`deploy.py` anchors its rate-limiter at WP3 and assumes `start.py` has already placed the arm there. Running `deploy.py` without `start.py` will cause large initial deltas that Tron2's safety layer silently drops.

## Checkpoint and model paths

| Thing | Path |
|---|---|
| Fine-tuned checkpoint | `/home/data/hf/hub/models--trantor2nd--tron2_gr00t_pick_step6k/snapshots/64108047dc22892c31f84856b507009578be5e79` |
| GR00T base model | `~/.cache/huggingface/hub/models--nvidia--GR00T-N1.5-3B/snapshots/<latest>` |
| Training dataset | `/home/hsb/TRON2_data/pick_stones` |

Override checkpoint with `TRON2_CKPT=<path>` and base model with `BASE_MODEL_PATH=<path>`.

`deploy.py` sets `HF_HUB_OFFLINE=1` so the model loads from local cache only — never attempts a network download.

## Key env vars

| Variable | Default | Effect |
|---|---|---|
| `TRON2_CKPT` | hardcoded snapshot path | fine-tuned checkpoint directory |
| `BASE_MODEL_PATH` | auto-detected from HF cache | nvidia/GR00T-N1.5-3B local path |
| `TRON2_CONSUME_STEPS` | `30` | how many steps of each K-step chunk to drive before re-inferring |

## Module map

```
tron2_ws.py        — WebSocket protocol layer: accid handshake, send_movej/send_gripper,
                     interp_send, warmup_sequence, shutdown_sequence, run(). Module-level
                     globals (ACCID, joint_values, gripper_values, should_exit) are the
                     shared state between start/deploy/shutdown.

observer.py        — ROS 2 node (Tron2Observer) that buffers the latest arm + gripper
                     JointState and three CompressedImage frames. Also: build_state_reorder
                     (maps live joint-name order → dataset schema), wait_for_fresh_observation.

inference.py       — GR00TRunner: loads checkpoint + pre/post pipelines, runs per-step
                     forward pass to reconstruct the full action chunk. build_stub_chunk:
                     synthetic sinusoidal chunk for control-path testing without GR00T.

controller.py      — WSController class: OOP alternative to tron2_ws module globals.
                     Has play_chunk(chunk, rate_hz). NOT used by deploy.py (which uses
                     tron2_ws directly); provided for standalone scripting.

deploy.py          — Entry point for live inference. Two-thread design: inference thread
                     calls GR00TRunner.infer, send thread plays the chunk at 10 Hz.
                     chunk_done event serializes them: send finishes → infer starts.

start.py           — Thin wrapper: connects WS, calls tron2_ws.warmup_sequence(), exits.
shutdown.py        — Thin wrapper: connects WS, calls tron2_ws.shutdown_sequence(), exits.
test.py            — Interactive keyboard joint controller (original reference script).
subscribe_and_viz.py — CV2 visualization of the three cameras + joint overlay. Run to
                       verify ROS topics are live before deploying.
test_infer_on_dataset.py — Offline MAE check: runs GR00T against stored dataset frames.
```

## Data flow

```
ROS 2 topics → Tron2Observer (observer.py)
                  /joint_states   → arm_pos (14 joints)
                  /gripper_state  → grip_pos (2 joints)
                  /camera/left/..., /camera/top/..., /camera/right/...

wait_for_fresh_observation() → (names, state[16], frames{3 cams}, stamp)

build_state_reorder(names) → reorder_idx  # align to dataset schema once at startup

state[reorder_idx][:16] → state16 (arm in rad, gripper in 0–1 after /100)

GR00TRunner.infer(left_wrist_bgr, cam_high_bgr, right_wrist_bgr, state16)
  → action chunk (K, 16)  # arm cols in rad, gripper cols in 0–1

deploy.py send thread: chunk[:CONSUME_STEPS]
  → sign-flip indices [0,1,2,3,5,6,8,9,13] for left-arm sign convention
  → tron2_ws.joint_values (14 floats)  +  gripper_values (×100 scale)
  → tron2_ws.send_movej() / send_gripper()  at SEND_INTERVAL=0.1 s
```

## Critical protocol details

- **accid**: never send a control frame until `wait_for_accid()` returns. The value comes from the first inbound WS message. `tron2_ws.run()` handles this automatically.
- **Left-arm sign flip**: joints at indices `[0,1,2,3,5,6,8,9,13]` are negated before sending `request_movej`. The training dataset and `/joint_states` use one convention; the robot's left-side movej uses the opposite.
- **Safety threshold**: Tron2 silently drops any `movej` whose per-joint delta from the last *commanded* value exceeds ~0.05 rad. `interp_send` and `step_toward` walk in `MAX_JOINT_STEP=0.1` rad increments (the robot tolerates slightly more than 0.05 in practice).
- **Gripper scaling**: dataset stores gripper opening in 0–1; robot protocol expects 0–100. Multiply by 100 before sending.
- **Dataset joint order**: `ARM_JOINT_NAMES` (14 joints) + `GRIPPER_NAMES` (2) defined in `observer.py:32`. Live `/joint_states` may arrive in a different order; `build_state_reorder` produces the remapping index.
- **Attention kernel**: always set `ATTENTION_IMPLEMENTATION=eager` (flash-attention not installed; eager matches the training environment).

## Offline inference smoke-test

```bash
conda activate lerobot_py310
python test_infer_on_dataset.py \
  --checkpoint <ckpt_path> \
  --dataset-root /home/hsb/TRON2_data/pick_stones \
  --indices 30,200,600,1200
```

Low MAE → inference pipeline correct. High MAE → pre/post/normalization broken.

## Visualization (verify ROS before inference)

```bash
conda activate lerobot_py310
python subscribe_and_viz.py   # press q to quit
```
