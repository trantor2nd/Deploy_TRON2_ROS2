#!/usr/bin/env python3
"""GR00T inference loop for Tron2.

Does NOT run any warmup. Assumes ``start.py`` has already walked the arm to
WP3 (see tron2_ws.WARMUP_WAYPOINT_3). Run order:

    python start.py     # once per power-on
    python deploy.py    # this file, until Ctrl+C
    python shutdown.py  # park the arm

Each cycle: read /joint_states + three CompressedImage frames, run GR00T,
then send every step of the returned action chunk directly as a movej —
no MAX_JOINT_STEP rate limiting. Note: Tron2's safety layer will silently
drop any movej whose per-joint delta exceeds ~0.05 rad from the previously
commanded value, so large jumps (e.g. WP3 → chunk[0]) may be ignored.

Left arm (j0..j6) is sign-flipped before going over the WebSocket: the training
dataset and /joint_states use one sign convention, request_movej for the left
side uses the opposite.

The WS plumbing (ACCID, ws_client, send_movej, etc.) is imported from
``tron2_ws``, so deploy.py / start.py / shutdown.py all speak the same
wire-level protocol as test.py.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional  # noqa: F401 (kept for future annotation use)

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


# ---- GR00T / ROS config ----
CHECKPOINT = Path(os.environ.get(
    "TRON2_CKPT",
    "/home/data/hf/hub/models--trantor2nd--tron2-pickup-gr00t/"
    "snapshots/007a769f46218f62e1d5a7e3fb3eb176716b03ce",
))
DEVICE = "cuda:0"
TASK_TEXT = "pick_up_stones_and_place_them_into_the_container"
BASE_MODEL_PATH = os.environ.get("BASE_MODEL_PATH")

JOINT_TOPIC = "/joint_states"
GRIPPER_TOPIC = "/gripper_state"
CAM_LEFT = "/camera/left/color/image_rect_raw/compressed"
CAM_HIGH = "/camera/top/color/image_raw/compressed"
CAM_RIGHT = "/camera/right/color/image_rect_raw/compressed"


def inference_task():
    log = logging.getLogger("infer")

    # --- 1. Wait for the WS handshake ---
    if not tron2_ws.wait_for_accid(timeout=15.0):
        print("[infer] timeout waiting for accid — aborting")
        tron2_ws.close()
        return
    print(f"[infer] ACCID acquired = {tron2_ws.ACCID}")

    # --- 2. Anchor the rate-limiter at WP3 (where start.py left the arm) ---
    #     This does NOT move the arm; it just tells step_toward() what the
    #     "previous commanded value" is so sub-send deltas are correct.
    tron2_ws.joint_values = list(tron2_ws.WARMUP_WAYPOINT_3)
    tron2_ws.gripper_values = [0.0, 0.0]
    print("[infer] rate-limiter anchored at WP3 (start.py must have run first)")
    tron2_ws.send_movej()
    time.sleep(tron2_ws.SEND_INTERVAL)

    # --- 3. Start ROS observer ---
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
    print("[infer] ROS observer spinning")

    # --- 4. Load GR00T ---
    runner = GR00TRunner(
        checkpoint=CHECKPOINT,
        device=DEVICE,
        task_text=TASK_TEXT,
        base_model_path=BASE_MODEL_PATH,
    )
    print("[infer] GR00T loaded")

    stop_event = threading.Event()

    # --- 5. Resolve joint-name ordering once ---
    print("[infer] waiting for first fresh observation…")
    obs = wait_for_fresh_observation(observer, log, stop_event)
    if obs is None:
        return
    names, state, frames, _ = obs
    reorder_idx = build_state_reorder(names)
    log_reorder_once(log, names, reorder_idx)
    if reorder_idx is None:
        print("[infer] joint name schema mismatch — aborting")
        return

    # --- 6. Main inference loop ---
    cycle = 0
    while not tron2_ws.should_exit:
        if cycle > 0:  # first cycle reuses `obs` captured above
            obs = wait_for_fresh_observation(observer, log, stop_event)
            if obs is None:
                break
            names, state, frames, _ = obs

        state16_raw = state.astype(np.float32)[reorder_idx][:16]
        state_for_model = state16_raw.copy()
        # /gripper_state is 0–100, dataset stored opening/100 → divide before model.
        state_for_model[14:16] = state_for_model[14:16] / 100.0

        cycle += 1
        arm_str = "[" + ",".join(f"{x:+.4f}" for x in state16_raw[:14]) + "]"
        print(f"\n[cycle {cycle}] STATE arm={arm_str} "
              f"grip=L{state16_raw[14]:.1f},R{state16_raw[15]:.1f}")

        t0 = time.monotonic()
        try:
            chunk = runner.infer(
                left_wrist_bgr=frames["left_wrist"][0],
                cam_high_bgr=frames["cam_high"][0],
                right_wrist_bgr=frames["right_wrist"][0],
                state16=state_for_model,
            )
        except Exception as exc:
            print(f"[infer] inference failed: {exc}")
            time.sleep(0.3)
            continue
        infer_ms = (time.monotonic() - t0) * 1000
        print(f"[cycle {cycle}] CHUNK K={len(chunk)} infer={infer_ms:.0f}ms")

        # Send each chunk step directly, no MAX_JOINT_STEP rate limiting.
        for k, cmd in enumerate(chunk):
            if tron2_ws.should_exit:
                return

            # Left arm (j0..j6) WS sign is opposite to dataset convention.
            target_joint = [float(x) for x in cmd[:14]]
            reverse_indices = [0, 1, 2, 3, 5, 6, 8, 9, 13]
            for idx in reverse_indices:
                target_joint[idx] = -target_joint[idx]

            if cmd.shape[0] >= 16:
                tron2_ws.gripper_values = [float(cmd[14]) * 100.0, float(cmd[15]) * 100.0]

            delta_now = max(abs(t - j) for t, j in zip(target_joint, tron2_ws.joint_values))
            tron2_ws.joint_values = target_joint
            tgt_str = "[" + ",".join(f"{x:+.4f}" for x in target_joint) + "]"
            print(f"[cycle {cycle}][step {k+1:2d}/{len(chunk)}] "
                  f"send={tgt_str} max|Δ|={delta_now:.3f} "
                  f"grip=L{tron2_ws.gripper_values[0]:.1f},R{tron2_ws.gripper_values[1]:.1f}")
            tron2_ws.send_movej()
            if cmd.shape[0] >= 16:
                tron2_ws.send_gripper()
            time.sleep(tron2_ws.SEND_INTERVAL)


if __name__ == "__main__":
    # Match oracle_realtime_deploy env: offline load + eager attention (the
    # kernel path the checkpoint was trained with).
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
