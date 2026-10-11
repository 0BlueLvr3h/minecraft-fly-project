r"""
Calibrazione della sterzata con gli odori (versione accurata)
-------------------------------------------------------------
Il primo test ha mostrato che il DNa02 sinistro spara quasi sempre più del destro,
qualunque sia l'odore. Qui cerchiamo se, SOTTO questo sbilanciamento, c'è
comunque un segnale giusto:

  per ogni candidato alla sterzata misuriamo (sinistra - destra) quando
  l'odore è a sinistra e quando è a destra, su più prove, e facciamo la differenza.
  Lo sbilanciamento fisso si cancella, resta solo l'effetto del lato.

  effetto > 0  ->  sterza VERSO l'odore      (giusto per 'fruity')
  effetto < 0  ->  sterza VIA dall'odore     (giusto per 'rotten')

Uso:
  python scripts\05_calibra_sterzata.py
  python scripts\05_calibra_sterzata.py --trials 8 --dur 1.0
"""

import argparse
import time

import numpy as np

from circuits import FlyCircuits
from flysim import FlyBrain

# Discendenti candidati per la sterzata (quelli che non esistono vengono saltati)
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
    ap.add_argument("--dur", type=float, default=1.0, help="secondi per prova")
    ap.add_argument("--trials", type=int, default=5, help="prove per condizione")
    ap.add_argument("--dist", type=float, default=1.0, help="distanza della sorgente (blocchi)")
    args = ap.parse_args()
    ms = args.dur * 1000

    brain = FlyBrain(dt=args.dt, seed=1)
    circ = FlyCircuits(brain)
    circ.set_noise(0)

    # gruppi da misurare: candidati + tutti i discendenti di un lato
    groups = {}
    for t in CANDIDATES:
        L, R = brain.find(t, "left"), brain.find(t, "right")
        if len(L) and len(R):
            groups[t] = {"L": L, "R": R}
    groups["TUTTI i DN"] = {"L": brain.find_superclass("descending", "left"),
                            "R": brain.find_superclass("descending", "right")}

    print("Gruppi misurati (neuroni sinistra + destra):")
    for name, g in groups.items():
        print(f"   {name:12s} {len(g['L'])} + {len(g['R'])}")
    for pre in ("DNa", "DNb"):
        found = brain.types_with_prefix(pre)
        print(f"   tipi {pre}*: " + ", ".join(found.index[:30]))

    t0 = time.perf_counter()
    effects = {}
    for odor in circ.odor:
        diffs = {name: [] for name in groups}            # effetto del lato, per prova
        for _ in range(args.trials):
            left = run_trial(brain, circ, [{"odor": odor, "angle": 90, "dist": args.dist}], ms, groups)
            right = run_trial(brain, circ, [{"odor": odor, "angle": -90, "dist": args.dist}], ms, groups)
            for name in groups:
                lat_left = left[name][0] - left[name][1]       # sinistra-destra, odore a sinistra
                lat_right = right[name][0] - right[name][1]    # sinistra-destra, odore a destra
                diffs[name].append((lat_left - lat_right) / 2)
        effects[odor] = diffs

    print(f"\nEffetto del lato dell'odore ({args.trials} prove, {args.dur} s ciascuna, "
          f"{time.perf_counter() - t0:.0f} s in tutto)")
    print("   + = sterza VERSO l'odore, - = sterza VIA.  Valori in Hz, ± errore standard.")
    print(f"   {'gruppo':12s} " + "".join(f"{o:>22s}" for o in effects))
    for name in groups:
        row = ""
        for odor in effects:
            d = np.array(effects[odor][name])
            se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
            mark = " *" if abs(d.mean()) > 2 * se and se > 0 else "  "
            row += f"{d.mean():+12.1f} ± {se:5.1f}{mark}"
        print(f"   {name:12s} {row}")
    print("\n   * = effetto più grande del doppio dell'errore (probabilmente reale)")
    print("   Ideale: 'fruity' positivo e 'rotten' negativo, entrambi con *.")


if __name__ == "__main__":
    main()
