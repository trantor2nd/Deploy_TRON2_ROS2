#!/usr/bin/env python3
"""Bring Tron2 from the fresh [0]*14 boot state to the inference pose (WP3).

Runs: anchor [0]*14 → WP1 → WP2 → WP3, each segment interpolated in ≤0.05 rad
steps so Tron2's safety layer engages the servo (same size step test.py uses
when you press 'q' or 'e'). Each landmark gets a hold pause so the arm can
physically arrive before the next leg starts. Exits after the final hold.

Run this once per power-on, before deploy.py:

    python start.py    # arm walks from [0]*14 to WP3, ~26 s
    python deploy.py   # inference loop
    python shutdown.py # reverse path, park arm at [0]*14
"""

import tron2_ws


def task():
    if not tron2_ws.wait_for_accid(timeout=15.0):
        print("[start] timeout waiting for accid — aborting")
        tron2_ws.close()
        return
    print(f"[start] ACCID acquired = {tron2_ws.ACCID}")
    tron2_ws.warmup_sequence()
    print("[start] arm at WP3 — closing connection")
    tron2_ws.close()


if __name__ == "__main__":
    tron2_ws.run(on_ready=task)
