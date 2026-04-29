#!/usr/bin/env python3
"""Shared Tron2 WebSocket plumbing for start.py / deploy.py / shutdown.py.

Keeps test.py's exact on-the-wire JSON shape and module-level globals:
``ACCID``, ``ws_client``, ``joint_values``, ``gripper_values``, ``MOVE_TIME``.
On top of that, exposes the three-waypoint trajectory used to bring the arm
safely in and out of the inference pose, plus the rate-limited step helper
that deploy.py uses to drive a model chunk.

Per-script usage:

    import tron2_ws

    def task():
        if not tron2_ws.wait_for_accid(15.0):
            return
        # ... do work, e.g. tron2_ws.warmup_sequence() ...
        tron2_ws.close()

    if __name__ == "__main__":
        tron2_ws.run(on_ready=task)

Every motion command goes through the same send_movej() / send_request()
path that test.py uses, so when something works in test.py it works here.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from typing import Callable, Optional, Sequence

import numpy as np
import websocket


# ---------------------------------------------------------------------------
# Protocol + motion constants (values chosen to match test.py exactly)
# ---------------------------------------------------------------------------

ROBOT_IP = "10.192.1.2"
MOVE_TIME = 0.2            # request_movej.time field, seconds
SEND_INTERVAL = 0.1        # pause between successive movej sends, seconds
MAX_JOINT_STEP = 0.1     # max per-send delta on any joint; Tron2 safety caps

# Hold duration at each warmup landmark (lets the hardware physically arrive).
WARMUP_HOLD_SECONDS = 3.0

# Three-waypoint safe path between [0]*14 and the inference-ready pose.
#   WP1: yaw_L/yaw_R opened symmetrically
#   WP2: yaws still open, knees folded
#   WP3: yaws returned to zero, knees folded — the operating pose
WARMUP_WAYPOINT_1 = [0.0, 0.23, 1.35, 0.0, 0.0, 0.0, 0.0,
                     0.0, -0.23, -1.35, 0.0, 0.0, 0.0, 0.0]
WARMUP_WAYPOINT_2 = [0.0, 0.23, 1.35, -1.6, 0.0, 0.0, 0.0,
                     0.0, -0.23, -1.35, -1.6, 0.0, 0.0, 0.0]
WARMUP_WAYPOINT_3 = [0.0, 0.23, 0.0, -1.6, 0.23, 0.0, 0.0,
                     0.0, -0.23, 0.0, -1.6, -0.23, 0.0, 0.0]


# ---------------------------------------------------------------------------
# Runtime state (module globals match test.py's top-level globals)
# ---------------------------------------------------------------------------

ACCID: Optional[str] = None
ws_client: Optional[websocket.WebSocketApp] = None
should_exit = False

joint_values = [0.0] * 14
gripper_values = [0.0, 0.0]

_accid_event = threading.Event()


# ---------------------------------------------------------------------------
# Wire protocol (identical shape to test.py.send_request)
# ---------------------------------------------------------------------------

def _generate_guid() -> str:
    return str(uuid.uuid4())


def send_request(title: str, data: Optional[dict] = None) -> None:
    """Exact port of test.py.send_request. Do not change the JSON shape."""
    if data is None:
        data = {}
    message = {
        "accid": ACCID,
        "title": title,
        "timestamp": int(time.time() * 1000),
        "guid": _generate_guid(),
        "data": data,
    }
    msg_str = json.dumps(message)
    print(f"\n[Send] {msg_str}")
    if ws_client is not None:
        ws_client.send(msg_str)
    else:
        print("[Error] ws_client is None")


def send_movej() -> None:
    send_request("request_movej", {
        "joint": joint_values,
        "time": MOVE_TIME,
    })


def send_gripper() -> None:
    send_request("request_set_limx_2fclaw_cmd", {
        "left_opening":  float(gripper_values[0]),
        "left_speed":    50.0,
        "left_force":    50.0,
        "right_opening": float(gripper_values[1]),
        "right_speed":   50.0,
        "right_force":   50.0,
    })


# ---------------------------------------------------------------------------
# Motion helpers (shared by warmup, shutdown, and the inference rate limiter)
# ---------------------------------------------------------------------------

def interp_send(from_pose: Sequence[float], to_pose: Sequence[float], label: str) -> None:
    """Step joint_values from ``from_pose`` to ``to_pose`` in ≤MAX_JOINT_STEP deltas.

    Each intermediate point is emitted via send_movej(). Tron2's safety layer
    only accepts the next movej when its delta from the previous commanded
    value stays under the threshold — matches test.py's 'q' increment.
    """
    global joint_values
    diff = [t - f for f, t in zip(from_pose, to_pose)]
    max_abs = max(abs(d) for d in diff)
    if max_abs < 1e-6:
        print(f"[{label}] from == to, skipping")
        return
    N = max(1, int(np.ceil(max_abs / MAX_JOINT_STEP)))
    print(f"[{label}] interp in {N} steps "
          f"(max|Δ|={max_abs:.3f} rad, ~{N * SEND_INTERVAL:.1f}s)")

    for k in range(1, N + 1):
        if should_exit:
            return
        alpha = k / N
        joint_values = [f + alpha * d for f, d in zip(from_pose, diff)]
        js = "[" + ",".join(f"{x:+.3f}" for x in joint_values) + "]"
        print(f"[{label}][{k:2d}/{N}] alpha={alpha:.3f} joint={js}")
        send_movej()
        time.sleep(SEND_INTERVAL)


def hold(seconds: float, label: str) -> None:
    """Sit on the last commanded pose for `seconds` so the arm can arrive."""
    if should_exit or seconds <= 0:
        return
    print(f"[{label}] holding commanded pose for {seconds:.1f}s")
    end = time.monotonic() + seconds
    while time.monotonic() < end and not should_exit:
        time.sleep(0.1)


def step_toward(target: Sequence[float], max_step: float) -> bool:
    """Advance joint_values toward target by at most max_step on any joint.

    Returns True when the target was reached this call. Used by deploy.py's
    inference loop to rate-limit every single movej out of the model chunk —
    the model might predict a first step far from the last commanded value
    (e.g., immediately after WP3), and the safety threshold requires we walk
    there in many small ≤MAX_JOINT_STEP sub-sends.
    """
    global joint_values
    diff = [t - j for t, j in zip(target, joint_values)]
    max_abs = max(abs(d) for d in diff) if diff else 0.0
    if max_abs <= max_step:
        joint_values = [float(t) for t in target]
        return True
    scale = max_step / max_abs
    joint_values = [j + scale * d for j, d in zip(joint_values, diff)]
    return False


# ---------------------------------------------------------------------------
# Full warmup / shutdown sequences
# ---------------------------------------------------------------------------

def warmup_sequence() -> None:
    """Bring the arm from [0]*14 to the inference pose via WP1 → WP2 → WP3."""
    global joint_values, gripper_values
    joint_values = [0.0] * 14
    gripper_values = [0.0, 0.0]
    print("[warmup] anchor commanded state at [0]*14")
    send_movej()
    time.sleep(SEND_INTERVAL)

    interp_send([0.0] * 14, WARMUP_WAYPOINT_1, "warmup-A")
    hold(WARMUP_HOLD_SECONDS, "warmup-A")
    interp_send(WARMUP_WAYPOINT_1, WARMUP_WAYPOINT_2, "warmup-B")
    hold(WARMUP_HOLD_SECONDS, "warmup-B")
    interp_send(WARMUP_WAYPOINT_2, WARMUP_WAYPOINT_3, "warmup-C")
    hold(WARMUP_HOLD_SECONDS, "warmup-C")
    print("[warmup] done — arm is at WP3")


def shutdown_sequence() -> None:
    """Reverse of warmup: WP3 → WP2 → WP1 → [0]*14."""
    global joint_values, gripper_values
    joint_values = list(WARMUP_WAYPOINT_3)
    print("[shutdown] anchor commanded state at WP3")
    send_movej()
    time.sleep(SEND_INTERVAL)

    interp_send(WARMUP_WAYPOINT_3, WARMUP_WAYPOINT_2, "shutdown-A")
    hold(WARMUP_HOLD_SECONDS, "shutdown-A")
    interp_send(WARMUP_WAYPOINT_2, WARMUP_WAYPOINT_1, "shutdown-B")
    hold(WARMUP_HOLD_SECONDS, "shutdown-B")
    interp_send(WARMUP_WAYPOINT_1, [0.0] * 14, "shutdown-C")
    hold(WARMUP_HOLD_SECONDS, "shutdown-C")
    print("[shutdown] done — arm parked at [0]*14")


# ---------------------------------------------------------------------------
# WS lifecycle
# ---------------------------------------------------------------------------

def _on_message(_ws, message: str) -> None:
    global ACCID
    try:
        root = json.loads(message)
        title = root.get("title", "")
        recv_accid = root.get("accid", None)
        if recv_accid is not None and ACCID is None:
            ACCID = recv_accid
            _accid_event.set()
        if title != "notify_robot_info":
            print(f"\n[Recv] {message}")
    except Exception as e:
        print(f"[Error] on_message parse failed: {e}")


def _on_error(_ws, error) -> None:
    print(f"[WebSocket Error] {error}")


def _on_close(_ws, _code, _msg) -> None:
    print("Connection closed.")


def wait_for_accid(timeout: float = 15.0) -> bool:
    """Block until the first inbound frame populates ACCID, or timeout."""
    return _accid_event.wait(timeout)


def close() -> None:
    """End the session: run() returns after ws_client.close() fires on_close."""
    global should_exit
    should_exit = True
    if ws_client is not None:
        try:
            ws_client.close()
        except Exception:
            pass


def run(on_ready: Callable[[], None], url: Optional[str] = None) -> None:
    """Connect, spawn ``on_ready`` in a daemon thread, block on run_forever.

    Call ``close()`` from within ``on_ready`` to exit cleanly.
    """
    global ws_client
    actual_url = url or f"ws://{ROBOT_IP}:5000"

    def _on_open(_ws):
        print("Connected!")
        threading.Thread(target=on_ready, daemon=True).start()

    ws_client = websocket.WebSocketApp(
        actual_url,
        on_open=_on_open,
        on_message=_on_message,
        on_error=_on_error,
        on_close=_on_close,
    )
    print(f"Connecting to {actual_url}  (Ctrl+C to quit)")
    ws_client.run_forever()
