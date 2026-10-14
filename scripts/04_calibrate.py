r"""
Calibration: spontaneous walking and olfaction
----------------------------------------------
Before hooking everything up to Minecraft, we check three things:

  1. the right neurons exist (walking, steering, antennae)
  2. with a bit of background noise the fly "walks" on its own, but without
     taking off at random
  3. an odor on one side makes the fly steer:
        fruit on the left    -> should turn LEFT  (moves towards it)
        rotten on the left   -> should turn RIGHT (avoids it)

Usage (from the project folder, with the venv active):
  python scripts\04_calibrate.py
  python scripts\04_calibrate.py --dur 3 --noise 0 2 5 10
"""

import argparse
import time

import numpy as np

from circuits import FlyCircuits
from flysim import FlyBrain

WINDOW_MS = 50.0


def simulate(brain, circ, ms, threats=None, odors=None):
    """Run the brain in windows like the server does, accumulating the results."""
    brain.reset()
    loom = circ.looming(threats)
    smell = circ.smell(odors)
    acc = {"fwd": 0.0, "back": 0.0, "turnL": 0.0, "turnR": 0.0}
    gf_spikes, takeoffs, n = 0, 0, 0
    t0 = time.perf_counter()
    for _ in range(int(ms / WINDOW_MS)):
        circ.apply(loom, smell)
        before = brain.counts.copy()
        brain.run(WINDOW_MS)
        out = circ.decode(brain.counts - before, WINDOW_MS)
        for k in acc:
            acc[k] += out["dn"][k]
        gf_spikes += sum(out["gf"])
        takeoffs += out["takeoff"]
        n += 1
    wall = time.perf_counter() - t0
    acc = {k: v / max(n, 1) for k, v in acc.items()}
    return acc, gf_spikes, takeoffs, (brain.counts > 0).sum(), ms / 1000 / wall


def steer(acc):
    tl, tr = acc["turnL"], acc["turnR"]
    if tl + tr == 0:
        return "no steering"
    idx = (tl - tr) / (tl + tr)
    side = "LEFT" if idx > 0 else "RIGHT"
    return f"towards {side} (index {idx:+.2f})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dt", type=float, default=0.2)
    ap.add_argument("--dur", type=float, default=2.0, help="seconds per trial")
    ap.add_argument("--noise", type=float, nargs="+", default=[0, 2, 5, 10],
                    help="background noise rates to try (Hz)")
    args = ap.parse_args()
    ms = args.dur * 1000

    t0 = time.perf_counter()
    brain = FlyBrain(dt=args.dt, seed=0)
    circ = FlyCircuits(brain)
    print(f"Brain loaded in {time.perf_counter() - t0:.1f} s\n")

    # 1. neurons found ----------------------------------------------------
    print("=== 1. Neurons found")
    lines, warnings = circ.report()
    for line in lines:
        print("   ", line)
    for w in warnings:
        print("    WARNING:", w)
    orn = brain.types_with_prefix("ORN")
    print(f"\n    Available ORN types ({len(orn)}):")
    print("    " + ", ".join(f"{k}({v})" for k, v in orn.items()))
    for pre in ("DNp09", "DNa02", "MDN", "DNg02"):
        found = brain.types_with_prefix(pre)
        print(f"    Types starting with {pre}: "
              + (", ".join(f"{k}({v})" for k, v in found.items()) or "none"))

    # 2. background noise -------------------------------------------------
    print("\n=== 2. Spontaneous activity (no stimulus, background noise only)")
    print("    noise      fwd     back   turn L   turn R    GF takeoff   active    rt")
    for hz in args.noise:
        circ.set_noise(hz)
        acc, gf, tk, active, rt = simulate(brain, circ, ms)
        print(f"    {hz:5.1f} Hz {acc['fwd']:7.1f} {acc['back']:8.1f} {acc['turnL']:8.1f}"
              f" {acc['turnR']:8.1f} {gf:5d} {tk:7d} {active:8,d} {rt:5.2f}")
    circ.set_noise(0)

    # 3. odors --------------------------------------------------------------
    print("\n=== 3. Odors (no noise), source at 1 block")
    print("    case                        fwd      back   turn L   turn R   result")
    for name in circ.odor:
        for label, angle in (("left", 90), ("right", -90)):
            odors = [{"odor": name, "angle": angle, "dist": 1.0}]
            acc, gf, tk, active, rt = simulate(brain, circ, ms, odors=odors)
            case = f"{name} on {label}"
            print(f"    {case:24s} {acc['fwd']:6.1f} {acc['back']:9.1f} {acc['turnL']:8.1f}"
                  f" {acc['turnR']:8.1f}   {steer(acc)}")

    print("\nPaste this output in the chat: we'll pick noise and thresholds from here.")


if __name__ == "__main__":
    main()
