r"""
Phase 1 - Validation test: escape reflex
----------------------------------------
Stimulates the "approaching object" detectors (LC4, LPLC2) and checks
whether the Giant Fiber (DNp01) fires. If it does, the escape circuit works
and we can hook it up to Minecraft.

Usage (from the project folder, with the venv active):
  python scripts\01_escape_test.py
  python scripts\01_escape_test.py --rate 100 --dur 0.5 --trials 3
"""

import argparse
import time

import numpy as np

from flysim import FlyBrain

CONDITIONS = [
    ("Control (no stimulus)",       []),
    ("LC4 both sides",              [("LC4", None)]),
    ("LC4 left only",               [("LC4", "left")]),
    ("LPLC2 both sides",            [("LPLC2", None)]),
    ("LC4 + LPLC2",                 [("LC4", None), ("LPLC2", None)]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=float, default=150, help="stimulus rate (Hz)")
    ap.add_argument("--dur", type=float, default=1.0, help="simulated seconds per trial")
    ap.add_argument("--trials", type=int, default=1, help="trials per condition")
    ap.add_argument("--top", type=int, default=10, help="number of most active neurons to show")
    args = ap.parse_args()

    t0 = time.perf_counter()
    brain = FlyBrain(seed=0)
    print(f"Brain loaded: {brain.N:,} neurons in {time.perf_counter() - t0:.1f} s\n")

    gf_l, gf_r = brain.find("DNp01", "left"), brain.find("DNp01", "right")

    for name, stims in CONDITIONS:
        idx = np.concatenate([brain.find(t, s) for t, s in stims]) if stims \
            else np.array([], np.int64)
        total = np.zeros(brain.N, np.float64)

        t0 = time.perf_counter()
        for _ in range(args.trials):
            brain.reset()
            brain.set_stimulus(idx, args.rate)
            brain.run(args.dur * 1000)
            total += brain.counts
        wall = time.perf_counter() - t0
        rate = total / (args.trials * args.dur)            # Hz

        print(f"=== {name}  ({len(idx)} neurons stimulated at {args.rate:.0f} Hz)")
        print(f"    Giant Fiber  left: {rate[gf_l].mean():6.1f} Hz   "
              f"right: {rate[gf_r].mean():6.1f} Hz")
        print(f"    Active neurons: {(rate > 0).sum():,}   "
              f"speed: {args.dur * args.trials / wall:.2f}x real time")

        others = rate.copy()
        others[idx] = 0
        top = np.argsort(others)[::-1][:args.top]
        top = [i for i in top if others[i] > 0]
        if top:
            print("    Most active (excluding stimulated ones):")
            for i in top:
                print(f"      {others[i]:6.1f} Hz  {brain.describe(i)}")
        print()


if __name__ == "__main__":
    main()
