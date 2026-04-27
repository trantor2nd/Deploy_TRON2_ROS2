#!/usr/bin/env python3
"""Tron2 observation layer.

Owns everything that turns ROS 2 traffic into the 16-DoF joint state + three
camera frames that GR00T expects. Also provides helpers to align the received
joint names to the training-dataset schema (the live publisher may use a
different order) and to block until a complete fresh observation is available.

Used by both the deploy entry point and the standalone `subscribe_and_viz`
tool, so the class lives here and every other module imports it.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage, JointState


# Dataset schema: observation.state columns are ARM_JOINT_NAMES + GRIPPER_NAMES
# in this exact order. Live /joint_states may publish in a different order;
# build_state_reorder maps back to this canonical form.
ARM_JOINT_NAMES = [
    "abad_L_Joint", "hip_L_Joint", "yaw_L_Joint", "knee_L_Joint",
    "wrist_yaw_L_Joint", "wrist_pitch_L_Joint", "wrist_roll_L_Joint",
    "abad_R_Joint", "hip_R_Joint", "yaw_R_Joint", "knee_R_Joint",
    "wrist_yaw_R_Joint", "wrist_pitch_R_Joint", "wrist_roll_R_Joint",
]
GRIPPER_NAMES = ["left_gripper", "right_gripper"]

# Camera slot names match dataset keys `observation.images.{slot}`.
CAM_SLOTS = ("left_wrist", "cam_high", "right_wrist")


@dataclass
class ImageSlot:
    topic: str
    frame: Optional[np.ndarray] = None
    stamp: float = 0.0


class Tron2Observer(Node):
    """ROS 2 node buffering the latest 14-DoF arm + 2-DoF gripper state and three cameras."""

    def __init__(self, joint_topic: str, gripper_topic: str, cam_topics: dict):
        super().__init__("tron2_observer")
        self._lock = threading.Lock()

        self.arm_name: List[str] = []
        self.arm_pos: Optional[np.ndarray] = None
        self.arm_stamp: float = 0.0
        self.grip_name: List[str] = []
        self.grip_pos: Optional[np.ndarray] = None
        self.grip_stamp: float = 0.0

        self.slots = {name: ImageSlot(topic=cam_topics[name]) for name in CAM_SLOTS}

        # Best-effort sensor QoS matches the robot's publishers for images/joints.
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.create_subscription(JointState, joint_topic, self._on_arm, sensor_qos)
        self.create_subscription(JointState, gripper_topic, self._on_grip, sensor_qos)
        for name in CAM_SLOTS:
            self.create_subscription(
                CompressedImage,
                cam_topics[name],
                lambda msg, n=name: self._on_image(msg, n),
                sensor_qos,
            )

        self.get_logger().info(f"subscribing arm: {joint_topic}")
        self.get_logger().info(f"subscribing gripper: {gripper_topic}")
        for name in CAM_SLOTS:
            self.get_logger().info(f"subscribing {name}: {cam_topics[name]}")

    # ------------------------------ callbacks ------------------------------
    def _on_arm(self, msg: JointState):
        pos = np.asarray(msg.position, dtype=np.float32) if msg.position else None
        with self._lock:
            self.arm_name = list(msg.name) if msg.name else self.arm_name
            self.arm_pos = pos
            self.arm_stamp = time.monotonic()

    def _on_grip(self, msg: JointState):
        pos = np.asarray(msg.position, dtype=np.float32) if msg.position else None
        with self._lock:
            self.grip_name = list(msg.name) if msg.name else self.grip_name
            self.grip_pos = pos
            self.grip_stamp = time.monotonic()

    def _on_image(self, msg: CompressedImage, slot_name: str):
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is None:
            self.get_logger().warn(f"imdecode failed on {slot_name} (format={msg.format})")
            return
        now = time.monotonic()
        with self._lock:
            slot = self.slots[slot_name]
            slot.frame = frame
            slot.stamp = now

    # ------------------------------- accessor ------------------------------
    def snapshot(self):
        """Return ((names, position, stamp) or None, {slot: (frame, stamp), ...})."""
        with self._lock:
            if self.arm_pos is None and self.grip_pos is None:
                joint = None
            else:
                names = list(self.arm_name) + list(self.grip_name)
                parts = []
                if self.arm_pos is not None:
                    parts.append(self.arm_pos)
                if self.grip_pos is not None:
                    parts.append(self.grip_pos)
                pos = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
                stamp = min(s for s in (self.arm_stamp, self.grip_stamp) if s > 0)
                joint = (names, pos, stamp)
            frames = {
                name: (None if s.frame is None else s.frame.copy(), s.stamp)
                for name, s in self.slots.items()
            }
        return joint, frames


# ---------------------------------------------------------------------------
# Joint-name alignment helpers
# ---------------------------------------------------------------------------

def build_state_reorder(observed_names: Sequence[str]) -> Optional[np.ndarray]:
    """Map ``observed_names`` -> canonical 16-dim order (ARM+GRIPPER).

    Returns an int64 index array so `state[idx]` yields the dataset-order
    state. Returns None if any expected name is missing (publisher is
    incomplete / schema mismatched).
    """
    expected = list(ARM_JOINT_NAMES) + list(GRIPPER_NAMES)
    name_to_idx = {n: i for i, n in enumerate(observed_names)}
    missing = [n for n in expected if n not in name_to_idx]
    if missing:
        return None
    return np.asarray([name_to_idx[n] for n in expected], dtype=np.int64)


def log_reorder_once(
    log: logging.Logger,
    names: Sequence[str],
    idx: Optional[np.ndarray],
) -> None:
    """One-shot diagnostic: did the live joint order need remapping?"""
    expected = list(ARM_JOINT_NAMES) + list(GRIPPER_NAMES)
    if idx is None:
        log.error(
            f"can't align to training schema — received names: {list(names)!r}; "
            f"expected: {expected!r}. Predictions will be garbage."
        )
    elif np.array_equal(idx, np.arange(len(expected))):
        log.info(f"joint names already in training order: {list(names)}")
    else:
        log.warning(
            f"REMAPPING joint order to training schema. "
            f"observed: {list(names)} -> reorder_idx: {idx.tolist()}"
        )


def wait_for_fresh_observation(
    observer: Tron2Observer,
    log: logging.Logger,
    stop_event: threading.Event,
    max_obs_age: float = 0.5,
    max_img_age: float = 0.2,
):
    """Block until all topics are populated and both joints and images are fresh.

    Returns ``(names, state, frames, joint_stamp)`` on success or ``None`` if
    ``stop_event`` fired or rclpy shut down while waiting. Logs a what-is-missing
    heartbeat every 2 s so the user can see which topic is lagging.

    Images are checked against ``max_img_age`` so the snapshot used for
    inference is temporally coherent — joints and cameras within the same
    window.
    """
    last_heartbeat = 0.0
    while not stop_event.is_set() and rclpy.ok():
        joint, frames = observer.snapshot()
        now = time.monotonic()
        missing = []
        if joint is None:
            missing.append("joint=<none>")
        else:
            _, state, _ = joint
            if len(state) < 16:
                missing.append(f"joint.dim={len(state)}<16")
        for n, (f, _) in frames.items():
            if f is None:
                missing.append(f"{n}=<no frame>")
        if missing:
            if now - last_heartbeat > 2.0:
                log.warning(f"waiting for observation: {', '.join(missing)}")
                last_heartbeat = now
            time.sleep(0.05)
            continue

        names, state, joint_stamp = joint
        age_joint = now - joint_stamp
        stale = age_joint > max_obs_age
        for n, (f, stamp) in frames.items():
            if now - stamp > max_img_age:
                stale = True
                break
        if stale:
            if now - last_heartbeat > 2.0:
                ages = f"joint={age_joint:.2f}s"
                for n2, (_, s2) in frames.items():
                    ages += f" {n2}={now-s2:.2f}s"
                log.warning(f"observation too stale: {ages}")
                last_heartbeat = now
            time.sleep(0.02)
            continue
        return names, state, frames, joint_stamp
    return None
