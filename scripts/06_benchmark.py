r"""
Benchmark: how fast the brain runs in game situations
-----------------------------------------------------
Measures the real-time factor (rt; above 1 = the fly reacts in time) in four
situations, for several simulation time steps (dt). This lets us pick the most
accurate dt that still keeps up with real time, even with an apple in hand.

It also checks that the escape still works with a larger dt (Giant Fiber).

Usage:
  pip install numba            (only once: fast engine)
  python scripts\06_benchmark.py
  python scripts\06_benchmark.py --dt 0.2 0.3 0.5
"""

import argparse
import time

import numpy as np

import flysim
from circuits import FlyCircuits
from flysim import FlyBrain

WINDOW_MS = 50.0
NONE = {"L": 0.0, "R": 0.0}


def scenario(circ, loom, odors):
    smell = {n: dict(NONE) for n in circ.odor}
    for name, level in odors.items():
        smell[name] = {"L": level, "R": level}
    return loom, smell


def measure(brain, circ, loom, smell, seconds):
    brain.reset()
    gf = 0
    t0 = time.perf_counter()
    for _ in range(int(seconds * 1000 / WINDOW_MS)):
        circ.apply(loom, smell)
        before = brain.counts.copy()
        brain.run(WINDOW_MS)
        gf += sum(circ.decode(brain.counts - before, WINDOW_MS)["gf"])
    wall = time.perf_counter() - t0
    return seconds / wall, int(brain.counts.sum() / seconds), gf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dt", type=float, nargs="+", default=[0.2, 0.3, 0.5])
    ap.add_argument("--sec", type=float, default=2.0, help="simulated seconds per situation")
    args = ap.parse_args()

    print(f"Engine: {'Numba' if flysim.HAVE_NUMBA else 'NumPy (install numba for the fast engine)'}")
    cases = [
        ("at rest",                 ({"L": 0.0, "R": 0.0}, {})),
        ("player running",          ({"L": 0.3, "R": 0.0}, {})),
        ("apple nearby",            (dict(NONE), {"fruity": 1.0})),
        ("apple a few blocks away", (dict(NONE), {"fruity": 0.3})),
    ]
    for dt in args.dt:
        brain = FlyBrain(dt=dt, seed=0)
        circ = FlyCircuits(brain)
        if flysim.HAVE_NUMBA:
            brain.run(1)                                   # compile (slow only the first time)
        print(f"\n=== dt = {dt} ms")
        print(f"    {'situation':24s} {'rt':>6s} {'spikes/s':>10s} {'Giant Fiber':>12s}")
        for name, (loom, odors) in cases:
            loom, smell = scenario(circ, loom, odors)
            rt, sps, gf = measure(brain, circ, loom, smell, args.sec)
            print(f"    {name:24s} {rt:6.2f} {sps:10,d} {gf:12d}")

    print("\nWe pick the smallest dt with rt > 1 in every row,"
          "\nas long as the Giant Fiber still fires with the running player.")


if __name__ == "__main__":
    main()
