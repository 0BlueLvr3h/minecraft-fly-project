r"""
Confronto finale delle configurazioni: olfatto, gusto e fuga insieme
---------------------------------------------------------------------
Ogni correzione al modello aggiusta qualcosa ma può romperne un'altra.
Qui ogni configurazione fa tutti gli esami:

  OLFATTO   frutta e marcio a 80 Hz: spike/s, tempo reale, decolli spuri,
            percezione (deve distinguere i due odori)
  GUSTO     lo zucchero di Shiu deve accendere MN9 (nell'originale: ~80 Hz)
  FUGA      distanza di decollo con un player che arriva correndo / accovacciato

Ora confronta la configurazione "fedele" (solo interneuroni corretti e taglio KC->KC)
con passi di simulazione diversi, e controlla se il cervello si calma dopo l'odore.

Uso:
  python scripts\12_confronto.py
"""

import time

import numpy as np

from circuits import FlyCircuits, _rate
from flysim import FlyBrain

WINDOW_MS = 50.0
NONE = {"L": 0.0, "R": 0.0}

SHIU_SUGAR = [
    720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345,
    720575940617000768, 720575940630797113, 720575940632889389, 720575940621754367,
    720575940621502051, 720575940640649691, 720575940639332736, 720575940616885538,
    720575940639198653, 720575940620900446, 720575940617937543, 720575940632425919,
    720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
    720575940611875570,
]
MN9_TYPE = "CB0701"

FEDELE = {"fix_ln_modulatory": True, "kc_recurrent": False}

# (nome, parametri, esenta anche il gusto dall'adattamento?, dt in ms)
CORRETTO = {**FEDELE, "modulators": "all"}     # i tre errori del connettoma corretti
CONFIGS = [
    ("corretto, senza adattamento",  CORRETTO, False, 0.2),
    ("corretto + adattamento 0.5",   {**CORRETTO, "adapt_b": 0.5, "tau_a": 300.0}, False, 0.2),
    ("corretto + adattamento 1",     {**CORRETTO, "adapt_b": 1.0, "tau_a": 300.0}, False, 0.2),
    ("corretto + adattamento 2",     {**CORRETTO, "adapt_b": 2.0, "tau_a": 300.0}, False, 0.2),
]


def windows(brain, circ, ms, loom=None, smell=None, stim=None):
    brain.reset()
    out = []
    for _ in range(int(ms / WINDOW_MS)):
        if stim is not None:
            brain.set_stimulus(*stim)
        else:
            circ.apply(loom or dict(NONE), smell or {n: dict(NONE) for n in circ.odor})
        c0 = brain.counts.copy()
        brain.run(WINDOW_MS)
        out.append(brain.counts - c0)
    return out


def approach(brain, circ, speed, start=12.0):
    brain.reset()
    dist = start
    while dist > 0.4:
        loom = circ.looming([{"angle": 30, "dist": dist, "speed": speed}])
        circ.apply(loom, {n: dict(NONE) for n in circ.odor})
        c0 = brain.counts.copy()
        brain.run(WINDOW_MS)
        d = brain.counts - c0
        if d[circ.gf["L"]].sum() + d[circ.gf["R"]].sum() > 0:
            return f"{dist:.1f}"
        dist -= speed * WINDOW_MS / 1000
    return "mai"


def main():
    for label, params, exempt_taste, dt in CONFIGS:
        t0 = time.perf_counter()
        brain = FlyBrain(dt=dt, seed=0, params=params)
        circ = FlyCircuits(brain)
        sugar = brain.neurons["idx"].to_numpy()[brain.neurons["root_id"].isin(SHIU_SUGAR).to_numpy()]
        mn9 = brain.find(MN9_TYPE)
        if exempt_taste:
            brain.exempt_adaptation(np.concatenate([sugar, mn9]))
        print(f"\n=== {label}   (neuromodulatori resi lenti: {brain.mod_silenced})")

        for name in ("fruity", "rotten"):
            for level, where in ((1.0, "vicino  "), (0.3, "lontano ")):
                smell = {n: dict(NONE) for n in circ.odor}
                smell[name] = {"L": level, "R": level}
                w0 = time.perf_counter()
                ws = windows(brain, circ, 1500, smell=smell)
                rt = 1.5 / (time.perf_counter() - w0)
                tot = sum(ws)
                gf = int(tot[circ.gf["L"]].sum() + tot[circ.gf["R"]].sum())
                pf = _rate(tot, circ.perceive_idx["fruity"], 1.5)
                pr = _rate(tot, circ.perceive_idx["rotten"], 1.5)
                # poi togliamo l'odore senza azzerare il cervello: si calma?
                after = []
                for _ in range(20):                                 # 1 s senza odore
                    circ.apply(dict(NONE), {n: dict(NONE) for n in circ.odor})
                    c0 = brain.counts.copy()
                    brain.run(WINDOW_MS)
                    after.append(int((brain.counts - c0).sum()))
                calm = sum(after[-6:]) / 0.3                        # spike/s nell'ultimo 0,3 s
                print(f"    {name:7s} {where} {tot.sum() / 1.5:9,.0f} spike/s  rt {rt:5.2f}  GF {gf:3d}  "
                      f"frutta {pf:4.0f} / marcio {pr:4.0f} Hz  | dopo 1 s senza odore: {calm:8,.0f} spike/s")

        ws = windows(brain, circ, 1000, stim=(sugar, 100.0))
        tot = sum(ws)
        print(f"    gusto   zucchero -> MN9 {_rate(tot, mn9, 1.0):5.1f} Hz   (originale di Shiu: ~80)")

        print(f"    fuga    decolla a: correndo {approach(brain, circ, 5.6)} blocchi, "
              f"accovacciato {approach(brain, circ, 1.3)} blocchi")
        print(f"    ({time.perf_counter() - t0:.0f} s)")

    print("\nCerchiamo la configurazione che passa tutti e tre gli esami.")


if __name__ == "__main__":
    main()
