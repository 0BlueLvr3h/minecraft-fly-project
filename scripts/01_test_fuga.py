r"""
Fase 1 - Test di validazione: riflesso di fuga
-----------------------------------------------
Stimola i rilevatori di "oggetto in avvicinamento" (LC4, LPLC2) e controlla
se il Giant Fiber (DNp01) spara. Se sì, il circuito di fuga funziona e
possiamo collegarlo a Minecraft.

Uso (dalla cartella del progetto, con il venv attivo):
  python scripts\01_test_fuga.py
  python scripts\01_test_fuga.py --rate 100 --dur 0.5 --trials 3
"""

import argparse
import time

import numpy as np

from flysim import FlyBrain

CONDITIONS = [
    ("Controllo (nessuno stimolo)", []),
    ("LC4 entrambi i lati",         [("LC4", None)]),
    ("LC4 solo sinistra",           [("LC4", "left")]),
    ("LPLC2 entrambi i lati",       [("LPLC2", None)]),
    ("LC4 + LPLC2",                 [("LC4", None), ("LPLC2", None)]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=float, default=150, help="Hz dello stimolo")
    ap.add_argument("--dur", type=float, default=1.0, help="secondi simulati per prova")
    ap.add_argument("--trials", type=int, default=1, help="prove per condizione")
    ap.add_argument("--top", type=int, default=10, help="neuroni più attivi da mostrare")
    args = ap.parse_args()

    t0 = time.perf_counter()
    brain = FlyBrain(seed=0)
    print(f"Cervello caricato: {brain.N:,} neuroni in {time.perf_counter() - t0:.1f} s\n")

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

        print(f"=== {name}  ({len(idx)} neuroni stimolati a {args.rate:.0f} Hz)")
        print(f"    Giant Fiber  sinistra: {rate[gf_l].mean():6.1f} Hz   "
              f"destra: {rate[gf_r].mean():6.1f} Hz")
        print(f"    Neuroni attivi: {(rate > 0).sum():,}   "
              f"velocità: {args.dur * args.trials / wall:.2f}x tempo reale")

        others = rate.copy()
        others[idx] = 0
        top = np.argsort(others)[::-1][:args.top]
        top = [i for i in top if others[i] > 0]
        if top:
            print("    Più attivi (esclusi gli stimolati):")
            for i in top:
                print(f"      {others[i]:6.1f} Hz  {brain.describe(i)}")
        print()


if __name__ == "__main__":
    main()
