r"""
Odor steering calibration (accurate version)
--------------------------------------------
The first test showed that the left DNa02 almost always fires more than the
right one, whatever the odor. Here we check whether, UNDERNEATH this
imbalance, there is still a correct signal:

  for every steering candidate we measure (left - right) when the odor is
  on the left and when it is on the right, over several trials, and take the
  difference. The fixed imbalance cancels out; only the effect of the side remains.

  effect > 0  ->  steers TOWARDS the odor     (correct for 'fruity')
  effect < 0  ->  steers AWAY from the odor   (correct for 'rotten')

Usage:
  python scripts\05_calibrate_steering.py
  python scripts\05_calibrate_steering.py --trials 8 --dur 1.0
"""

import argparse
import time

import numpy as np

from circuits import FlyCircuits
from flysim import FlyBrain

# Candidate descending neurons for steering (the ones that don't exist are skipped)
CANDIDATES = ["DNa01", "DNa02", "DNa03", "DNa04", "DNb05", "DNb06", "DNg13", "DNp09", "MDN"]


def run_trial(brain, circ, odors, ms, groups):
    brain.reset()
    circ.apply(circ.looming([]), circ.smell(odors))
    brain.run(ms)
    s = ms / 1000
    return {name: (brain.counts[g["L"]].sum() / max(len(g["L"]), 1) / s,
                   brain.counts[g["R"]].sum() / max(len(g["R"]), 1) / s)
            for name, g in groups.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dt", type=float, default=0.2)
    ap.add_argument("--dur", type=float, default=1.0, help="seconds per trial")
    ap.add_argument("--trials", type=int, default=5, help="trials per condition")
    ap.add_argument("--dist", type=float, default=1.0, help="source distance (blocks)")
    args = ap.parse_args()
    ms = args.dur * 1000

    brain = FlyBrain(dt=args.dt, seed=1)
    circ = FlyCircuits(brain)
    circ.set_noise(0)

    # groups to measure: candidates + all descending neurons of one side
    groups = {}
    for t in CANDIDATES:
        L, R = brain.find(t, "left"), brain.find(t, "right")
        if len(L) and len(R):
            groups[t] = {"L": L, "R": R}
    groups["ALL DNs"] = {"L": brain.find_superclass("descending", "left"),
                         "R": brain.find_superclass("descending", "right")}

    print("Measured groups (left + right neurons):")
    for name, g in groups.items():
        print(f"   {name:12s} {len(g['L'])} + {len(g['R'])}")
    for pre in ("DNa", "DNb"):
        found = brain.types_with_prefix(pre)
        print(f"   {pre}* types: " + ", ".join(found.index[:30]))

    t0 = time.perf_counter()
    effects = {}
    for odor in circ.odor:
        diffs = {name: [] for name in groups}            # effect of the side, per trial
        for _ in range(args.trials):
            left = run_trial(brain, circ, [{"odor": odor, "angle": 90, "dist": args.dist}], ms, groups)
            right = run_trial(brain, circ, [{"odor": odor, "angle": -90, "dist": args.dist}], ms, groups)
            for name in groups:
                lat_left = left[name][0] - left[name][1]       # left-right, odor on the left
                lat_right = right[name][0] - right[name][1]    # left-right, odor on the right
                diffs[name].append((lat_left - lat_right) / 2)
        effects[odor] = diffs

    print(f"\nEffect of the odor side ({args.trials} trials, {args.dur} s each, "
          f"{time.perf_counter() - t0:.0f} s in total)")
    print("   + = steers TOWARDS the odor, - = steers AWAY.  Values in Hz, ± standard error.")
    print(f"   {'group':12s} " + "".join(f"{o:>22s}" for o in effects))
    for name in groups:
        row = ""
        for odor in effects:
            d = np.array(effects[odor][name])
            se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
            mark = " *" if abs(d.mean()) > 2 * se and se > 0 else "  "
            row += f"{d.mean():+12.1f} ± {se:5.1f}{mark}"
        print(f"   {name:12s} {row}")
    print("\n   * = effect larger than twice the standard error (probably real)")
    print("   Ideal: 'fruity' positive and 'rotten' negative, both with *.")


if __name__ == "__main__":
    main()
