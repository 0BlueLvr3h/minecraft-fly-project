r"""
Benchmark: quanto è veloce il cervello nelle situazioni di gioco
-----------------------------------------------------------------
Misura il tempo reale (rt, sopra 1 = la mosca reagisce in tempo) in quattro
situazioni, per diversi passi di simulazione (dt). Così scegliamo il dt più
preciso che regge il tempo reale anche con la mela in mano.

Controlla anche che con un dt più grosso la fuga funzioni ancora (Giant Fiber).

Uso:
  pip install numba            (una volta sola: motore veloce)
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
    ap.add_argument("--sec", type=float, default=2.0, help="secondi simulati per situazione")
    args = ap.parse_args()

    print(f"Motore: {'Numba' if flysim.HAVE_NUMBA else 'NumPy (installa numba per il motore veloce)'}")
    cases = [
        ("a riposo",                ({"L": 0.0, "R": 0.0}, {})),
        ("player che corre",        ({"L": 0.3, "R": 0.0}, {})),
        ("mela vicina",             (dict(NONE), {"fruity": 1.0})),
        ("mela a qualche blocco",   (dict(NONE), {"fruity": 0.3})),
    ]
    for dt in args.dt:
        brain = FlyBrain(dt=dt, seed=0)
        circ = FlyCircuits(brain)
        if flysim.HAVE_NUMBA:
            brain.run(1)                                   # compila (solo la prima volta è lento)
        print(f"\n=== dt = {dt} ms")
        print(f"    {'situazione':24s} {'rt':>6s} {'spike/s':>10s} {'Giant Fiber':>12s}")
        for name, (loom, odors) in cases:
            loom, smell = scenario(circ, loom, odors)
            rt, sps, gf = measure(brain, circ, loom, smell, args.sec)
            print(f"    {name:24s} {rt:6.2f} {sps:10,d} {gf:12d}")

    print("\nScegliamo il dt più piccolo con rt > 1 in tutte le righe,"
          "\npurché il Giant Fiber spari ancora con il player che corre.")


if __name__ == "__main__":
    main()
