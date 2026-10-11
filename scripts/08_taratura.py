r"""
Taratura di olfatto e fuga (con la configurazione del server)
--------------------------------------------------------------
Dai registri di gioco sono emersi due problemi:
  - l'odore accende troppo il cervello: rallenta, fa decollare la mosca senza
    motivo e confonde frutta e marcio
  - la fuga scatta troppo presto, anche avvicinandosi accovacciati

Parte A: prova diverse intensità dell'odore (Hz sugli ORN). Per ognuna:
         spike, tempo reale, decolli spuri, e quanto si distinguono i due odori.
Parte B: prova diverse soglie di looming. Per ognuna simula un player che
         arriva da 12 blocchi accovacciato, camminando e correndo, e misura a che
         distanza la mosca decolla.

Uso:
  python scripts\08_taratura.py              (configurazione consigliata)
  python scripts\08_taratura.py --minimal    (solo correzione interneuroni)
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
SPEEDS = [("accovacciato", 1.3), ("camminando", 4.3), ("correndo", 5.6)]
TRIALS = 3


# Solo la correzione degli interneuroni: il modello più vicino all'originale di Shiu
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
    """Player che arriva dritto: distanza al primo decollo (None = mai)."""
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
                    help="solo la correzione degli interneuroni (senza adattamento né taglio KC->KC)")
    args = ap.parse_args()
    params = MINIMAL if args.minimal else RECOMMENDED
    print(f"Configurazione: {'MINIMA' if args.minimal else 'consigliata'} {params}\n")
    brain, circ = brain_like_server(params)

    print("=== A. Intensità dell'odore")
    print("    (percezione in Hz dei neuroni di frutta / marcio: per distinguerli,")
    print("     con la frutta deve salire quella della frutta e restare bassa quella del marcio)")
    print(f"    {'Hz ORN':>6s}  {'odore':8s} {'livello':>7s} {'spike/s':>10s} {'rt':>5s} {'GF':>4s}"
          f" {'perc.frutta':>12s} {'perc.marcio':>12s}")
    for rate in ODOR_RATES:
        circuits.ODOR_RATE = rate
        for name in ("fruity", "rotten"):
            for level in (1.0, 0.3):
                sps, rt, gf, pf, pr = odor_test(brain, circ, name, level)
                print(f"    {rate:6d}  {name:8s} {level:7.1f} {sps:10,.0f} {rt:5.2f} {gf:4d}"
                      f" {pf:12.0f} {pr:12.0f}")
    circuits.ODOR_RATE = 80.0

    print("\n=== B. Soglia del looming: distanza a cui decolla (blocchi), media di "
          f"{TRIALS} prove; '-' = non decolla")
    print(f"    {'soglia':>7s}  " + "".join(f"{lbl:>15s}" for lbl, _ in SPEEDS))
    for lm in LOOM_MINS:
        circuits.LOOM_MIN = lm
        row = ""
        for _, sp in SPEEDS:
            hits = [approach(brain, circ, sp) for _ in range(TRIALS)]
            got = [h for h in hits if h is not None]
            row += f"{(np.mean(got) if got else float('nan')):12.1f} {len(got)}/{TRIALS}" if got \
                else f"{'-':>12s} 0/{TRIALS}"
        print(f"    {lm:7.2f}  {row}")

    print("\nCerchiamo: A) l'intensità più alta con GF = 0, rt > 1 e frutta/marcio ben distinti;")
    print("           B) la soglia per cui accovacciati ci si avvicina molto più che correndo.")


if __name__ == "__main__":
    main()
