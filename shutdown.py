#!/usr/bin/env python3
"""Park Tron2 back to [0]*14 by reversing the warmup trajectory.

Assumes the arm is currently at or close to WP3 (i.e., start.py + deploy.py
just ran). Walks WP3 → WP2 → WP1 → [0]*14 in ≤0.05 rad steps with the same
safety profile as start.py.

    python start.py     # once per power-on
    python deploy.py    # inference until Ctrl+C
    python shutdown.py  # then park the arm
"""

import tron2_ws


def task():
    if not tron2_ws.wait_for_accid(timeout=15.0):
        print("[shutdown] timeout waiting for accid — aborting")
        tron2_ws.close()
        return
    print(f"[shutdown] ACCID acquired = {tron2_ws.ACCID}")
    tron2_ws.shutdown_sequence()
    print("[shutdown] arm parked — closing connection")
    tron2_ws.close()


if __name__ == "__main__":
    tron2_ws.run(on_ready=task)
