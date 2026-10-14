r"""
Olfaction and escape tuning (with the server configuration)
-----------------------------------------------------------
Two problems emerged from the game logs:
  - the odor excites the brain too much: it slows down, makes the fly take off
    for no reason and confuses fruit with rotten
  - the escape triggers too early, even when approaching while crouching

Part A: tries several odor intensities (Hz on the ORNs). For each one:
        spikes, real-time factor, spurious takeoffs, and how well the two odors
        are told apart.
Part B: tries several looming thresholds. For each one simulates a player
        approaching from 12 blocks while crouching, walking and sprinting, and
        measures at what distance the fly takes off.

Usage:
  python scripts\08_tuning.py              (recommended configuration)
  python scripts\08_tuning.py --minimal    (interneuron fix only)
"""

import time

import numpy as np

import circuits
from circuits import FlyCircuits, _rate
from flysim import RECOMMENDED, FlyBrain

WINDOW_MS = 50.0
NONE = {"L": 0.0, "R": 0.0}
ODOR_RATES = [10, 20, 40, 80]
LOOM_MINS = [0.0, 0.15, 0.25, 0.35, 0.5]
SPEEDS = [("crouching", 1.3), ("walking", 4.3), ("sprinting", 5.6)]
TRIALS = 3


# Interneuron fix only: the model closest to Shiu's original
MINIMAL = dict(fix_ln_modulatory=True)


def brain_like_server(params):
    brain = FlyBrain(dt=0.2, seed=0, params=params)
    return brain, FlyCircuits(brain)


def odor_test(brain, circ, name, level, seconds=1.5):
    smell = {n: dict(NONE) for n in circ.odor}
    smell[name] = {"L": level, "R": level}
    brain.reset()
    gf, spikes, pf, pr, n = 0, 0, 0.0, 0.0, 0
    t0 = time.perf_counter()
    for _ in range(int(seconds * 1000 / WINDOW_MS)):
        circ.apply(dict(NONE), smell)
        c0 = brain.counts.copy()
        brain.run(WINDOW_MS)
        d = brain.counts - c0
        gf += int(d[circ.gf["L"]].sum() + d[circ.gf["R"]].sum())
        spikes += int(d.sum())
        pf += _rate(d, circ.perceive_idx["fruity"], WINDOW_MS / 1000)
        pr += _rate(d, circ.perceive_idx["rotten"], WINDOW_MS / 1000)
        n += 1
    rt = seconds / (time.perf_counter() - t0)
    return spikes / seconds, rt, gf, pf / n, pr / n


def approach(brain, circ, speed, start=12.0, angle=30.0):
    """Player approaching in a straight line: distance at first takeoff (None = never)."""
    brain.reset()
    dist = start
    while dist > 0.4:
        loom = circ.looming([{"angle": angle, "dist": dist, "speed": speed}])
        circ.apply(loom, {n: dict(NONE) for n in circ.odor})
        c0 = brain.counts.copy()
        brain.run(WINDOW_MS)
        d = brain.counts - c0
        if d[circ.gf["L"]].sum() + d[circ.gf["R"]].sum() > 0:
            return dist
        dist -= speed * WINDOW_MS / 1000
    return None


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--minimal", action="store_true",
                    help="interneuron fix only (no adaptation, no KC->KC cut)")
    args = ap.parse_args()
    params = MINIMAL if args.minimal else RECOMMENDED
    print(f"Configuration: {'MINIMAL' if args.minimal else 'recommended'} {params}\n")
    brain, circ = brain_like_server(params)

    print("=== A. Odor intensity")
    print("    (perception in Hz of the fruit / rotten neurons: to tell them apart,")
    print("     with fruit the fruit one must rise while the rotten one stays low)")
    print(f"    {'Hz ORN':>6s}  {'odor':8s} {'level':>7s} {'spikes/s':>10s} {'rt':>5s} {'GF':>4s}"
          f" {'fruit perc.':>12s} {'rotten perc.':>12s}")
    for rate in ODOR_RATES:
        circuits.ODOR_RATE = rate
        for name in ("fruity", "rotten"):
            for level in (1.0, 0.3):
                sps, rt, gf, pf, pr = odor_test(brain, circ, name, level)
                print(f"    {rate:6d}  {name:8s} {level:7.1f} {sps:10,.0f} {rt:5.2f} {gf:4d}"
                      f" {pf:12.0f} {pr:12.0f}")
    circuits.ODOR_RATE = 80.0

    print("\n=== B. Looming threshold: takeoff distance (blocks), mean of "
          f"{TRIALS} trials; '-' = no takeoff")
    print(f"    {'threshold':>9s}" + "".join(f"{lbl:>15s}" for lbl, _ in SPEEDS))
    for lm in LOOM_MINS:
        circuits.LOOM_MIN = lm
        row = ""
        for _, sp in SPEEDS:
            hits = [approach(brain, circ, sp) for _ in range(TRIALS)]
            got = [h for h in hits if h is not None]
            row += f"{(np.mean(got) if got else float('nan')):12.1f} {len(got)}/{TRIALS}" if got \
                else f"{'-':>12s} 0/{TRIALS}"
        print(f"    {lm:7.2f}  {row}")

    print("\nWe're looking for: A) the highest intensity with GF = 0, rt > 1 and fruit/rotten clearly distinct;")
    print("                   B) the threshold at which a crouching player gets much closer than a sprinting one.")


if __name__ == "__main__":
    main()
