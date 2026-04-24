# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status

Greenfield repo — only `goal.md`, `LICENSE`, `README.md` exist. No source, build system, or tests yet. The first implementation work still needs to be scaffolded. Read `goal.md` (in Chinese) for the full task brief before writing code.

## Project goal

Deploy the **Tron2** robot by running **GR00T** inference on-device and driving the arm over WebSocket:

1. Load GR00T weights trained with LeRobot (HuggingFace repo `trantor2nd/tron2-pickup-gr00t`; dataset `trantor2nd/tron2_lerobot_pickup`).
2. Subscribe via **ROS 2** to the 14-DoF joint state and three camera streams. The expected observation layout should mirror the LeRobot dataset schema.
3. Run GR00T to produce an **action chunk** (sequence of 14-DoF targets).
4. Stream the chunk to the robot over **WebSocket at ~10 Hz** (one `request_movej` per tick).

Training was done in a conda env named `lerobot_py310`. Inference likely needs the same env (or a compatible one) on the deploy machine.

## Robot control interface (from `goal.md`)

- Transport: `ws://10.192.1.2:5000` using the `websocket-client` package.
- Every outgoing message is JSON with fields: `accid`, `title`, `timestamp` (ms), `guid` (uuid4), `data`.
- **`accid` is not known at connect time** — it is learned from the first inbound message (any frame that carries `accid`; `notify_robot_info` is the common one). Do not send control frames before `accid` has been populated.
- Motion command: `title="request_movej"`, `data={"joint": [14 floats], "time": <seconds>}`. The reference script uses `time=0.2` per step, which is consistent with a 10 Hz stream (5 Hz would need 0.2 s, 10 Hz needs ~0.1 s — re-check against GR00T's action-chunk cadence before picking a value).
- Joint vector is length 14; order must match whatever the LeRobot dataset uses (verify before sending real commands).

When building the deploy loop, keep the send thread decoupled from inference so a slow model step does not stall the 10 Hz control cadence. Buffer the latest action chunk and let the sender consume it at a fixed rate.

## Conventions to establish on first implementation

When you add the first real code, decide and then record here:
- The Python env/lockfile strategy (conda `environment.yml` vs. `pyproject.toml`).
- The ROS 2 distro and workspace layout (`colcon` package, node names, topic names for joint state + three cameras).
- How checkpoints are fetched (HF Hub download vs. local path) and where they are cached.
- How to run the deploy node end-to-end (single command to launch ROS 2 node + WebSocket client).

Until those are chosen, this file intentionally does not list build/test/run commands — there are none yet.
