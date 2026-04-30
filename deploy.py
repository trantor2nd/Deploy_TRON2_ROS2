#!/usr/bin/env python3
"""GR00T inference loop for Tron2.

Assumes start.py has already walked the arm to WP3. Run order:

    python start.py     # once per power-on
    python deploy.py    # this file, until Ctrl+C
    python shutdown.py  # park the arm

Each cycle (sequential, no threads):
    1. Capture a fresh, synchronized observation (joint state + 3 cameras).
    2. Run GR00T → action chunk (K, 16).
    3. Send the first CONSUME_STEPS steps at 10 Hz, blocking until done.
    4. Go to 1 — observe the state AFTER the chunk fully executed.

Left arm joints are sign-flipped before sending (dataset and /joint_states use
one convention; request_movej for the left side uses the opposite).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor

import tron2_ws
from observer import (
    Tron2Observer,
    build_state_reorder,
    log_reorder_once,
    wait_for_fresh_observation,
)
from inference import GR00TRunner


# ── config ────────────────────────────────────────────────────────────────────

CHECKPOINT = Path(os.environ.get(
    "TRON2_CKPT",
    "/home/data/hf/hub/models--trantor2nd--tron2_gr00t_pick_step6k/"
    "snapshots/64108047dc22892c31f84856b507009578be5e79",
))
DEVICE = "cuda:0"
TASK_TEXT = "pick_up_stones_and_place_them_into_the_container"
BASE_MODEL_PATH = os.environ.get("BASE_MODEL_PATH")
# Steps of each K-step chunk to actually drive; the rest are discarded and
# re-inferred from the state observed after the chunk completes.
CONSUME_STEPS = int(os.environ.get("TRON2_CONSUME_STEPS", "30"))

JOINT_TOPIC = "/joint_states"
GRIPPER_TOPIC = "/gripper_state"
CAM_LEFT = "/camera/left/color/image_rect_raw/compressed"
CAM_HIGH = "/camera/top/color/image_raw/compressed"
CAM_RIGHT = "/camera/right/color/image_rect_raw/compressed"

# Left-arm joint indices whose sign convention differs between the dataset
# and the robot's request_movej protocol.
_LEFT_FLIP_IDX = [0, 1, 2, 3, 5, 6, 8, 9, 13]


# ── helpers ───────────────────────────────────────────────────────────────────

def _send_step(cmd: np.ndarray) -> None:
    """Apply one (16,) action row: sign-flip left arm, then send movej + gripper."""
    joints = [float(x) for x in cmd[:14]]
    for i in _LEFT_FLIP_IDX:
        joints[i] = -joints[i]
    tron2_ws.joint_values = joints
    tron2_ws.send_movej()
    if cmd.shape[0] >= 16:
        tron2_ws.gripper_values = [float(cmd[14]) * 100.0, float(cmd[15]) * 100.0]
        tron2_ws.send_gripper()


def _execute_chunk(chunk: np.ndarray, log: logging.Logger, cycle: int) -> None:
    """Send each row of chunk at SEND_INTERVAL Hz, blocking until done."""
    next_tick = time.monotonic()
    for k, cmd in enumerate(chunk):
        if tron2_ws.should_exit:
            break
        sleep = next_tick - time.monotonic()
        if sleep > 0:
            time.sleep(sleep)
        _send_step(cmd)
        joints_str = "[" + ",".join(f"{x:+.4f}" for x in tron2_ws.joint_values) + "]"
        log.info(
            f"[cycle {cycle}][{k+1:2d}/{len(chunk)}] "
            f"joint={joints_str} "
            f"grip=L{tron2_ws.gripper_values[0]:.1f},R{tron2_ws.gripper_values[1]:.1f}"
        )
        next_tick += tron2_ws.SEND_INTERVAL


# ── main task ─────────────────────────────────────────────────────────────────

def inference_task() -> None:
    log = logging.getLogger("deploy")

    # 1. Wait for WebSocket handshake.
    if not tron2_ws.wait_for_accid(timeout=15.0):
        log.error("timeout waiting for accid — aborting")
        tron2_ws.close()
        return
    log.info(f"ACCID = {tron2_ws.ACCID}")

    # 2. Anchor the rate-limiter at WP3 (start.py must have run first).
    #    This does NOT move the arm; it tells send_movej what the previous
    #    commanded value is so per-send deltas are computed correctly.
    tron2_ws.joint_values = list(tron2_ws.WARMUP_WAYPOINT_3)
    tron2_ws.gripper_values = [0.97 * 100.0, 0.0]
    tron2_ws.send_movej()
    time.sleep(tron2_ws.SEND_INTERVAL)
    tron2_ws.send_gripper()
    time.sleep(tron2_ws.SEND_INTERVAL)

    # 3. Start ROS observer.
    rclpy.init()
    observer = Tron2Observer(
        joint_topic=JOINT_TOPIC,
        gripper_topic=GRIPPER_TOPIC,
        cam_topics={
            "left_wrist": CAM_LEFT,
            "cam_high": CAM_HIGH,
            "right_wrist": CAM_RIGHT,
        },
    )
    executor = MultiThreadedExecutor()
    executor.add_node(observer)
    threading.Thread(target=executor.spin, daemon=True).start()
    log.info("ROS observer spinning")

    # 4. Load GR00T.
    runner = GR00TRunner(
        checkpoint=CHECKPOINT,
        device=DEVICE,
        task_text=TASK_TEXT,
        base_model_path=BASE_MODEL_PATH,
    )
    log.info("GR00T loaded")

    # 5. Resolve joint-name ordering once from the first live observation.
    stop_event = threading.Event()  # never set; wait_for_fresh_observation also exits on rclpy shutdown
    log.info("waiting for first observation…")
    obs = wait_for_fresh_observation(observer, log, stop_event)
    if obs is None:
        return
    names, _, _, _ = obs
    reorder_idx = build_state_reorder(names)
    log_reorder_once(log, names, reorder_idx)
    if reorder_idx is None:
        log.error("joint name schema mismatch — aborting")
        return

    # 6. Main loop: observe → infer → execute → repeat.
    cycle = 0
    while not tron2_ws.should_exit:
        cycle += 1

        # ── Observe ──────────────────────────────────────────────────────────
        obs = wait_for_fresh_observation(observer, log, stop_event)
        if obs is None:
            break
        names, state, frames, _ = obs

        state16_raw = state.astype(np.float32)[reorder_idx][:16]
        # Model expects arm in rad (already), gripper in 0–1 (divide robot's 0–100).
        state_for_model = state16_raw.copy()
        state_for_model[14:16] /= 100.0

        arm_str = "[" + ",".join(f"{x:+.4f}" for x in state16_raw[:14]) + "]"
        log.info(f"[cycle {cycle}] STATE arm={arm_str} grip=L{state16_raw[14]:.1f},R{state16_raw[15]:.1f}")

        # ── Infer ─────────────────────────────────────────────────────────────
        t0 = time.monotonic()
        try:
            chunk = runner.infer(
                left_wrist_bgr=frames["left_wrist"][0],
                cam_high_bgr=frames["cam_high"][0],
                right_wrist_bgr=frames["right_wrist"][0],
                state16=state_for_model,
            )
        except Exception as exc:
            log.error(f"inference failed: {exc}")
            continue
        log.info(f"[cycle {cycle}] CHUNK K={len(chunk)} infer={1000*(time.monotonic()-t0):.0f}ms")

        # ── Execute ───────────────────────────────────────────────────────────
        drive_chunk = chunk[:CONSUME_STEPS]
        log.info(f"[cycle {cycle}] executing {len(drive_chunk)}/{len(chunk)} steps")
        _execute_chunk(drive_chunk, log, cycle)


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_USE_FLASH_ATTENTION_2", "0")
    os.environ.setdefault("USE_FLASH_ATTENTION", "0")
    os.environ.setdefault("FLASH_ATTENTION_SKIP_CUDA_BUILD", "1")
    os.environ.setdefault("ATTENTION_IMPLEMENTATION", "eager")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    tron2_ws.run(on_ready=inference_task)
