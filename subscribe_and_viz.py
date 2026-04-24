#!/usr/bin/env python3
"""Subscribe Tron2 observation topics and visualize them.

Uses `observer.Tron2Observer` as the subscription layer (same class the deploy
pipeline uses), and renders the three RGB cameras side-by-side with the
joint-state overlay. Handy to verify the ROS side before launching inference.
"""

import argparse
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor

from observer import Tron2Observer, CAM_SLOTS

IMG_H, IMG_W = 480, 640


def _placeholder(text: str) -> np.ndarray:
    img = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
    cv2.putText(
        img, text, (30, IMG_H // 2),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 80), 2, cv2.LINE_AA,
    )
    return img


def _fit(frame: np.ndarray) -> np.ndarray:
    if frame.shape[0] != IMG_H or frame.shape[1] != IMG_W:
        return cv2.resize(frame, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    return frame


def _overlay_joint(canvas: np.ndarray, joint, age: float):
    x, y = 12, 22
    line_h = 16
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    color = (255, 255, 255)
    shadow = (0, 0, 0)

    def put(text, pos):
        cv2.putText(canvas, text, (pos[0] + 1, pos[1] + 1), font, scale, shadow, 2, cv2.LINE_AA)
        cv2.putText(canvas, text, pos, font, scale, color, 1, cv2.LINE_AA)

    if joint is None:
        put("joint_state: <no data>", (x, y))
        return

    names, pos, _ = joint
    put(f"joint_state  dim={len(pos)}  age={age * 1000:6.0f} ms", (x, y))
    y += line_h + 2
    for i, v in enumerate(pos):
        label = names[i] if i < len(names) else f"j{i:02d}"
        put(f"{i:2d} {label:<22s} {v:+8.4f}", (x, y))
        y += line_h


def _overlay_cam_age(canvas: np.ndarray, text: str, origin):
    cv2.rectangle(
        canvas,
        (origin[0] - 4, origin[1] - 16),
        (origin[0] + 180, origin[1] + 6),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        canvas, text, origin,
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,
    )


def run_viz(node: Tron2Observer, fps: float) -> int:
    period = 1.0 / fps
    window = "Tron2 observation (q to quit)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, IMG_W * len(CAM_SLOTS) // 2, IMG_H // 2)
    next_tick = time.monotonic()
    try:
        while rclpy.ok():
            now = time.monotonic()
            joint, frames = node.snapshot()

            tiles = []
            for name in CAM_SLOTS:
                frame, stamp = frames[name]
                if frame is None:
                    tile = _placeholder(f"{name}: no frame")
                else:
                    tile = _fit(frame).copy()
                    age_ms = (now - stamp) * 1000 if stamp else -1
                    _overlay_cam_age(tile, f"{name}  {age_ms:5.0f} ms", (10, IMG_H - 14))
                tiles.append(tile)

            canvas = np.concatenate(tiles, axis=1)
            joint_age = now - joint[2] if joint else 0.0
            _overlay_joint(canvas, joint, joint_age)

            cv2.imshow(window, canvas)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                return 0

            next_tick += period
            sleep = next_tick - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_tick = time.monotonic()  # fell behind, resync
    finally:
        cv2.destroyAllWindows()
    return 0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--joint-topic", default="/joint_states", help="14-DoF arm JointState topic")
    p.add_argument("--gripper-topic", default="/gripper_state", help="2-DoF gripper JointState topic")
    p.add_argument(
        "--cam-left",
        default="/camera/left/color/image_rect_raw/compressed",
        help="left wrist CompressedImage (column 1)",
    )
    p.add_argument(
        "--cam-high",
        default="/camera/top/color/image_raw/compressed",
        help="top/high CompressedImage (column 2)",
    )
    p.add_argument(
        "--cam-right",
        default="/camera/right/color/image_rect_raw/compressed",
        help="right wrist CompressedImage (column 3)",
    )
    p.add_argument("--fps", type=float, default=10.0, help="viz refresh rate")
    return p.parse_args()


def main():
    args = parse_args()
    cam_topics = {
        "left_wrist": args.cam_left,
        "cam_high": args.cam_high,
        "right_wrist": args.cam_right,
    }

    rclpy.init()
    node = Tron2Observer(args.joint_topic, args.gripper_topic, cam_topics)
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        rc = run_viz(node, args.fps)
    except KeyboardInterrupt:
        rc = 0
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
