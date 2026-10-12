r"""
Condizionamento: la mosca impara chi sei?
------------------------------------------
L'esperimento classico sulle Drosophila, fatto sul connettoma con memory.py, con le
stesse finestre da 50 ms del gioco e con gli odori dei player scelti da
17_odori_player.py (lancialo prima).

Tre player:
  A  le dà da mangiare: odore A insieme alla ricompensa (dopaminergici PAM)
  B  la colpisce: odore B con tre colpi (dopaminergici PPL1)
  C  un estraneo: non le ha mai fatto niente

Poi annusa A, B e C da soli. Valenza imparata: 0 = neutro, > 0 = buono (ci va
incontro), < 0 = cattivo (lo evita). Ci aspettiamo A positivo, B negativo, C vicino
a 0. L'esperimento è ripetuto più volte: il cervello è rumoroso come quello vero.
Alla fine un controllo: la frutta da sola, senza rinforzo, non deve insegnare niente.

Uso:
  python scripts\16_condizionamento.py
"""

import json
import time

import numpy as np

from circuits import ODOR_RATE, ODORS, FlyCircuits, player_rate
from flysim import PROC, RECOMMENDED, FlyBrain
from memory import MushroomBody

WIN = 50.0
NEAR, FAR = 0.6, 0.3      # concentrazione: circa 2 e 4 blocchi
REPEATS = 3


def odor(brain, glomeruli, profile):
    idx = [brain.find(g) for g in glomeruli]
    hz = [np.full(len(i), w) for i, w in zip(idx, profile)]
    return np.concatenate(idx), np.concatenate(hz)


def run(brain, mb, orn, seconds, conc=NEAR, reward=0.0, punish=0.0, fixed_hz=None):
    """Simula `seconds` secondi a finestre da 50 ms; restituisce la valenza media dell'ultimo secondo."""
    vals = []
    n = int(seconds * 1000 / WIN)
    for k in range(n):
        if orn is None:
            brain.clear_stimulus()
        else:
            brain.set_stimulus(orn[0], (fixed_hz or player_rate(conc)) * orn[1])
        for idx, hz in mb.reinforcement(reward, punish):
            brain.add_stimulus(idx, hz)
        before = brain.counts.copy()
        brain.run(WIN)
        mb.update(brain.counts - before, WIN)
        if k >= n - int(1000 / WIN):
            vals.append(mb.valence)
    return float(np.mean(vals))


def main():
    t0 = time.perf_counter()
    f = PROC / "player_odors.json"
    if not f.exists():
        raise SystemExit("Manca data/processed/player_odors.json: lancia prima scripts\\17_odori_player.py")
    odors = json.loads(f.read_text(encoding="utf-8"))
    brain = FlyBrain(dt=0.2, seed=3, params=RECOMMENDED)
    FlyCircuits(brain)
    mb = MushroomBody(brain)
    a, b, c = (odor(brain, o["glomeruli"], o["profile"]) for o in odors[:3])
    none = None
    print(f"Pronto in {time.perf_counter() - t0:.1f} s: {len(mb.pos):,} sinapsi KC->MBON, "
          f"MBON letti: {(mb.comp == 1).sum()} PAM, {(mb.comp == -1).sum()} PPL1")

    rows = []
    for rep in range(REPEATS):
        mb.reset()
        brain.reset()
        r = {"A prima": run(brain, mb, a, 1.5), "B prima": run(brain, mb, b, 1.5)}
        run(brain, mb, a, 6, reward=1.0)                      # A le dà da mangiare
        run(brain, mb, none, 1)
        for _ in range(3):                                    # B la colpisce tre volte
            run(brain, mb, b, 0.5)
            run(brain, mb, b, 1.0, punish=1.0)
        run(brain, mb, none, 1)
        r.update({"A": run(brain, mb, a, 2), "B": run(brain, mb, b, 2), "C estraneo": run(brain, mb, c, 2),
                  "A lontano": run(brain, mb, a, 2, conc=FAR), "B lontano": run(brain, mb, b, 2, conc=FAR)})
        rows.append(r)
        print(f"  prova {rep + 1}: " + "  ".join(f"{k} {v:+.2f}" for k, v in r.items()))

    print("\nMedia delle prove (valenza: + buono, - cattivo):")
    for k in rows[0]:
        v = [r[k] for r in rows]
        print(f"  {k:12s} {np.mean(v):+.2f}   (da {min(v):+.2f} a {max(v):+.2f})")

    mb.reset()
    brain.reset()
    fruit = np.concatenate([brain.find(g) for g in ODORS["fruity"]])
    fruit = (fruit, np.ones(len(fruit)))
    run(brain, mb, fruit, 10, fixed_hz=ODOR_RATE)
    run(brain, mb, none, 1)
    print(f"\nControllo, 10 s di frutta senza rinforzo: frutta {run(brain, mb, fruit, 2, fixed_hz=ODOR_RATE):+.2f}, "
          f"A {run(brain, mb, a, 2):+.2f}  (devono restare a 0)")
    print(f"\nFatto in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
