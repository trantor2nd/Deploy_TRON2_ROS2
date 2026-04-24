#!/usr/bin/env python3
"""Tron2 robot control layer (WebSocket protocol).

One class, ``WSController``: opens the ws://host:5000 connection, learns the
``accid`` from the first inbound frame, and exposes synchronous send helpers:

  - ``send_movej``  → request_movej              (14 joint targets + time)
  - ``send_gripper``→ request_set_limx_2fclaw_cmd (left/right opening 0–100)
  - ``play_chunk``  → iterates an (K, 16) action array at ``rate_hz`` and
                      fires one movej + gripper per tick, logging each step.

No background send thread — the caller drives cadence, keeping inference and
actuation strictly serialized.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from typing import Optional, Sequence
from urllib.parse import urlparse

import numpy as np
import websocket


class WSController:
    """Tron2 websocket controller: accid handshake + synchronous send helpers."""

    def __init__(
        self,
        url: str,
        move_time: float = 0.5,
        log_notify_seconds: float = 5.0,
    ) -> None:
        self.url = url
        self.move_time = move_time
        self.log = logging.getLogger("ws")

        self._accid: Optional[str] = None
        self._accid_event = threading.Event()
        self._ws: Optional[websocket.WebSocketApp] = None
        self._send_lock = threading.Lock()
        self._running = False
        self._ws_thread: Optional[threading.Thread] = None

        # Temporarily log `notify_robot_info` for this many seconds after
        # handshake, so we can read state flags like `enabled`, `mode`, etc.
        # Afterwards those notifies are dropped as usual.
        self._notify_log_until: float = 0.0
        self._log_notify_seconds = log_notify_seconds

    # --------------------------- lifecycle ---------------------------------
    def start(self) -> None:
        self._running = True
        # Robot is on the LAN; any inherited HTTP(S)_PROXY will black-hole the
        # connection. websocket-client reads env vars before its own no_proxy
        # arg, so scrub them here instead.
        for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
                  "all_proxy", "ALL_PROXY"):
            stripped = os.environ.pop(k, None)
            if stripped:
                self.log.info(f"unset proxy env {k}={stripped}")
        host = urlparse(self.url).hostname or ""
        if host:
            os.environ["NO_PROXY"] = (os.environ.get("NO_PROXY", "") + "," + host).lstrip(",")

        self._ws = websocket.WebSocketApp(
            self.url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        self._ws_thread = threading.Thread(target=self._ws.run_forever, daemon=True)
        self._ws_thread.start()

    def stop(self) -> None:
        self._running = False
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass

    def wait_ready(self, timeout: float = 10.0) -> bool:
        """Block until the first inbound frame populates ``accid``."""
        return self._accid_event.wait(timeout)

    # --------------------------- ws callbacks ------------------------------
    # websocket-client passes the WebSocketApp as the first positional arg to
    # every callback. We don't need it — `_ws` prefix keeps lint quiet.
    def _on_open(self, _ws):
        self.log.info(f"connected to {self.url}")

    def _on_message(self, _ws, message: str):
        try:
            payload = json.loads(message)
        except Exception:
            return
        accid = payload.get("accid")
        if accid and self._accid is None:
            self._accid = accid
            self._accid_event.set()
            self.log.info(f"acquired accid={accid}")
            # Start the notify-info sniffing window now that we're handshaked.
            self._notify_log_until = time.monotonic() + self._log_notify_seconds

        title = payload.get("title", "")
        if title == "notify_robot_info":
            # Sniff state during the initial window: prints enable/mode/error
            # fields that tell us whether the arm is powered on. After the
            # window expires these notifies are dropped (one per tick → spam).
            if time.monotonic() < self._notify_log_until:
                self.log.info(
                    f"[notify] {json.dumps(payload.get('data'), ensure_ascii=False)[:400]}"
                )
        else:
            self.log.info(
                f"[recv] {title}: {json.dumps(payload.get('data'), ensure_ascii=False)[:300]}"
            )

    def _on_error(self, _ws, error):
        self.log.warning(f"ws error: {error}")

    def _on_close(self, _ws, code, msg):
        self.log.info(f"ws closed code={code} msg={msg}")

    # --------------------------- send path ---------------------------------
    def _send(self, title: str, data: dict) -> None:
        if self._accid is None or self._ws is None:
            return
        frame = {
            "accid": self._accid,
            "title": title,
            "timestamp": int(time.time() * 1000),
            "guid": str(uuid.uuid4()),
            "data": data,
        }
        with self._send_lock:
            try:
                self._ws.send(json.dumps(frame))
            except Exception as exc:
                self.log.warning(f"send failed: {exc}")

    def send_movej(self, joint14: Sequence[float]) -> None:
        self._send("request_movej", {"joint": list(map(float, joint14)), "time": self.move_time})

    def warmup_hold(
        self,
        joint14: Sequence[float],
        repeats: int = 3,
        interval: float = 0.3,
    ) -> None:
        """Send `movej(joint14)` a few times to engage the arm servo safely.

        Many industrial-grade arms keep the servo inactive after power-on and
        only latch it when the FIRST movej arrives with a target that is very
        close to the current joint state (safety: prevent big initial jerks).
        After that, arbitrary movej commands are accepted.

        Calling this with the *currently observed* joint state effectively
        asks the arm to "hold your pose", which is the safest way to trigger
        that first-movej latch. Matches what test.py accidentally does when
        its initial joint_values=[0]*14 happens to be near the current pose.
        """
        if self._accid is None:
            self.log.warning("warmup_hold: no accid yet, skipping")
            return
        js = "[" + ",".join(f"{float(x):+.4f}" for x in joint14) + "]"
        self.log.info(
            f"warmup: sending {repeats}x movej(hold) joint={js} time={self.move_time}s "
            f"— engages servo without moving the arm"
        )
        for i in range(repeats):
            self._send("request_movej", {"joint": list(map(float, joint14)),
                                         "time": self.move_time})
            time.sleep(interval)

    def send_gripper(
        self,
        left_opening: float,
        right_opening: float,
        speed: float = 50.0,
        force: float = 50.0,
    ) -> None:
        """Send a two-finger gripper opening command.

        All four numeric fields use the Limx protocol's 0–100 scale. Caller is
        responsible for rescaling from the policy's 0–1 convention.
        """
        clamp = lambda v: max(0.0, min(100.0, float(v)))
        self._send("request_set_limx_2fclaw_cmd", {
            "left_opening":  clamp(left_opening),
            "left_speed":    clamp(speed),
            "left_force":    clamp(force),
            "right_opening": clamp(right_opening),
            "right_speed":   clamp(speed),
            "right_force":   clamp(force),
        })

    # --------------------------- chunk playback ----------------------------
    def play_chunk(
        self,
        chunk: np.ndarray,
        rate_hz: float,
        stop_event: Optional[threading.Event] = None,
    ) -> int:
        """Fire movej (+ gripper) for each row of ``chunk`` at ``rate_hz``.

        Blocks until the chunk is fully sent or ``stop_event`` is set. Returns
        the number of steps actually transmitted. Each step is logged with the
        exact joint vector being sent so motion failures can be cross-checked
        against the commands ``test.py`` used to drive the arm manually.
        """
        if chunk is None or len(chunk) == 0:
            return 0
        dt = 1.0 / max(rate_hz, 1e-3)
        sent = 0
        next_tick = time.monotonic()
        for k, cmd in enumerate(chunk):
            if stop_event is not None and stop_event.is_set():
                break

            joint14 = [float(x) for x in cmd[:14]]
            self.send_movej(joint14)
            joint_str = "[" + ",".join(f"{x:+.4f}" for x in joint14) + "]"

            grip_str = ""
            if cmd.shape[0] >= 16:
                # Dataset stores gripper as opening/100 (0.0–0.98); scale up for protocol.
                left_grip = float(cmd[14]) * 100.0
                right_grip = float(cmd[15]) * 100.0
                self.send_gripper(left_grip, right_grip)
                grip_str = f" | grip L={left_grip:5.1f} R={right_grip:5.1f}"

            self.log.info(
                f"step {k+1:2d}/{len(chunk)} movej t={self.move_time:.2f}s "
                f"joint={joint_str}{grip_str}"
            )

            sent = k + 1
            next_tick += dt
            sleep = next_tick - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_tick = time.monotonic()  # fell behind; resync
        return sent
