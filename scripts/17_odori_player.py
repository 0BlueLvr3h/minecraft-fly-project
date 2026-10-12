r"""
Odori dei player: quali sa distinguere il cervello?
----------------------------------------------------
La mosca riconosce un player solo dall'odore, e lo impara nelle cellule di Kenyon (KC).
Perché ogni player resti "lui", il suo odore deve:
  - accendere abbastanza KC (sennò la memoria ha poco su cui scrivere)
  - accendere sempre le stesse (sennò ogni volta sembra un odore nuovo)
  - sovrapporsi poco con gli odori degli altri player (sennò li confonde)

Generiamo tanti odori candidati (un gruppo di glomeruli, ognuno con la sua intensità),
li facciamo annusare al cervello due volte a circa 2 blocchi di distanza e teniamo i
migliori. Il cervello non viene toccato: scegliamo solo gli odori.

Scrive data/processed/player_odors.json, che il server usa per dare un odore a ogni
player. Da rifare solo se cambia il connettoma o la configurazione del cervello.

Uso:
  python scripts\17_odori_player.py
"""

import json
import time

import numpy as np

from circuits import ODOR_RATE, ODORS, FlyCircuits, player_rate
from flysim import PROC, RECOMMENDED, FlyBrain

CANDIDATES = 120
KEEP = 16                 # quanti odori diversi (oltre: due player possono condividerne uno)
CONC = 0.6                # circa un player a 2 blocchi
MS = 600.0
MAX_OVERLAP = 0.12        # sovrapposizione massima tra le KC di due odori scelti
MIN_KC = 40               # KC affidabili minime perché un odore valga
# Glomeruli esclusi: già usati per frutta e marcio, o con un significato innato forte (feromoni)
RESERVED = {g for gl in ODORS.values() for g in gl} | {"ORN_DA1", "ORN_VA1d", "ORN_VA1v", "ORN_DL3"}
OUT = PROC / "player_odors.json"


def main():
    t0 = time.perf_counter()
    brain = FlyBrain(dt=0.2, seed=17, params=RECOMMENDED)
    FlyCircuits(brain)
    kc = brain.find_prefix("KC")
    ct = brain.neurons["cell_type"].fillna("").astype(str)
    pool = sorted(t for t in ct[ct.str.startswith("ORN_")].unique() if t not in RESERVED)
    print(f"Cervello pronto in {time.perf_counter() - t0:.1f} s; {len(pool)} glomeruli disponibili, "
          f"{CANDIDATES} odori da provare")

    rng = np.random.default_rng(2024)
    cands = []
    for c in range(CANDIDATES):
        n = int(rng.integers(10, 19))
        gl = list(rng.choice(pool, n, replace=False))
        prof = [round(float(x), 2) for x in rng.uniform(0.3, 1.0, n)]
        idx = np.concatenate([brain.find(g) for g in gl])
        hz = np.concatenate([np.full(len(brain.find(g)), w) for g, w in zip(gl, prof)])
        sets = []
        for _ in range(2):
            brain.reset()
            brain.set_stimulus(idx, player_rate(CONC) * hz)
            brain.run(MS)
            sets.append(set(kc[brain.counts[kc] >= 2].tolist()))
        reliable = sets[0] & sets[1]
        union = sets[0] | sets[1]
        cands.append({"glomeruli": gl, "profile": prof, "kc": reliable,
                      "reliability": len(reliable) / max(1, len(union))})
        if (c + 1) % 20 == 0:
            print(f"  {c + 1}/{CANDIDATES} ({time.perf_counter() - t0:.0f} s)")

    good = [c for c in cands if len(c["kc"]) >= MIN_KC]
    print(f"\nOdori che accendono almeno {MIN_KC} KC affidabili: {len(good)} su {CANDIDATES}")
    good.sort(key=lambda c: len(c["kc"]) * c["reliability"], reverse=True)

    chosen = []
    for c in good:
        ov = max((len(c["kc"] & d["kc"]) / len(c["kc"] | d["kc"]) for d in chosen), default=0.0)
        if ov <= MAX_OVERLAP:
            c["overlap"] = ov
            chosen.append(c)
        if len(chosen) == KEEP:
            break

    print(f"Scelti {len(chosen)} odori:")
    for i, c in enumerate(chosen):
        print(f"  {i:2d}: {len(c['glomeruli']):2d} glomeruli, KC affidabili {len(c['kc']):4d} "
              f"({100 * len(c['kc']) / len(kc):.1f}%), ripetibilità {c['reliability']:.2f}, "
              f"sovrapposizione max {c['overlap']:.2f}")
    if len(chosen) < KEEP:
        print(f"ATTENZIONE: solo {len(chosen)} odori abbastanza distinti, i player oltre li condivideranno")

    OUT.write_text(json.dumps([{"glomeruli": c["glomeruli"], "profile": c["profile"]} for c in chosen],
                              indent=1), encoding="utf-8")
    print(f"\nSalvato {OUT} in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
