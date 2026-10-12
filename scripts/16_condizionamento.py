r"""
Condizionamento: la mosca impara un odore?
-------------------------------------------
L'esperimento classico sulle Drosophila, fatto sul connettoma con memory.py e con le
stesse finestre da 50 ms del gioco:

  RICOMPENSA  odore A da solo, poi A insieme alla ricompensa (come se la mosca
              mangiasse vicino a un player), poi A e B (di controllo) da soli
  PUNIZIONE   memoria azzerata; odore A con due colpi brevi, poi A e B da soli

Valenza imparata: 0 = odore neutro, > 0 = buono (ci va incontro), < 0 = cattivo.
Ci aspettiamo: dopo la ricompensa A positivo, dopo la punizione A negativo, B
sempre vicino a 0.

Uso:
  python scripts\16_condizionamento.py
"""

import time

import numpy as np

from circuits import ODOR_RATE, ODORS, FlyCircuits
from flysim import RECOMMENDED, FlyBrain
from memory import MushroomBody

WIN = 50.0
CONC = 0.6                # concentrazione dell'odore: circa un player a 2 blocchi
GLOMERULI = 15            # un odore di un corpo accende molti glomeruli, a intensità diverse
HALF = 0.3                # i recettori saturano: a questa concentrazione danno già metà risposta
RESERVED = {g for gl in ODORS.values() for g in gl} | {"ORN_DA1", "ORN_VA1d", "ORN_VA1v", "ORN_DL3"}


def odor(brain, glomeruli, profile):
    idx = [brain.find(g) for g in glomeruli]
    hz = [np.full(len(i), w) for i, w in zip(idx, profile)]
    return np.concatenate(idx), np.concatenate(hz)


def run(brain, mb, orn, seconds, reward=0.0, punish=0.0):
    """Simula `seconds` secondi a finestre da 50 ms; restituisce la valenza media dell'ultimo secondo."""
    vals, fam = [], []
    n = int(seconds * 1000 / WIN)
    resp = CONC / (CONC + HALF) * (1 + HALF)               # risposta sublineare dei recettori
    for k in range(n):
        if orn is None:
            brain.clear_stimulus()
        else:
            brain.set_stimulus(orn[0], ODOR_RATE * resp * orn[1])
        for idx, hz in mb.reinforcement(reward, punish):
            brain.add_stimulus(idx, hz)
        before = brain.counts.copy()
        brain.run(WIN)
        mb.update(brain.counts - before, WIN)
        if k >= n - int(1000 / WIN):
            vals.append(mb.valence)
            fam.append(mb.familiar)
    return float(np.mean(vals)), float(np.mean(fam))


def show(label, v):
    print(f"  {label:34s} valenza {v[0]:+.2f}   (odore sentito {v[1]:.2f})")


def main():
    t0 = time.perf_counter()
    brain = FlyBrain(dt=0.2, seed=3, params=RECOMMENDED)
    FlyCircuits(brain)
    mb = MushroomBody(brain)
    ct = brain.neurons["cell_type"].fillna("").astype(str)
    print(f"Pronto in {time.perf_counter() - t0:.1f} s: {len(mb.pos):,} sinapsi KC->MBON, "
          f"MBON letti: {(mb.comp == 1).sum()} PAM, {(mb.comp == -1).sum()} PPL1")

    pool = sorted(t for t in ct[ct.str.startswith("ORN_")].unique() if t not in RESERVED)
    rng = np.random.default_rng(11)
    gl = [rng.choice(pool, GLOMERULI, replace=False) for _ in range(2)]
    orn_a, orn_b = (odor(brain, x, rng.uniform(0.3, 1.0, GLOMERULI)) for x in gl)
    none = None
    shared = len(set(gl[0]) & set(gl[1]))
    print("Odore A:", " ".join(g[4:] for g in gl[0]))
    print("Odore B:", " ".join(g[4:] for g in gl[1]), f"  ({shared} glomeruli in comune con A)")

    print("\nRICOMPENSA")
    show("A prima", run(brain, mb, orn_a, 2))
    show("B prima", run(brain, mb, orn_b, 2))
    run(brain, mb, orn_a, 4, reward=1.0)
    run(brain, mb, none, 1)
    show("A dopo 4 s di A + ricompensa", run(brain, mb, orn_a, 2))
    show("B dopo", run(brain, mb, orn_b, 2))
    print(f"  memoria cambiata: {100 * mb.strength():.1f}% dei pesi KC->MBON")

    mb.reset()
    brain.reset()
    print("\nPUNIZIONE")
    show("A prima", run(brain, mb, orn_a, 2))
    for _ in range(2):
        run(brain, mb, orn_a, 0.5)
        run(brain, mb, orn_a, 0.5, punish=1.0)
    run(brain, mb, none, 1)
    show("A dopo 2 colpi con A", run(brain, mb, orn_a, 2))
    show("B dopo", run(brain, mb, orn_b, 2))
    print(f"  memoria cambiata: {100 * mb.strength():.1f}% dei pesi KC->MBON")

    print("\nDISIMPARARE: dopo la punizione, A + ricompensa per 6 s")
    run(brain, mb, orn_a, 6, reward=1.0)
    run(brain, mb, none, 1)
    show("A dopo", run(brain, mb, orn_a, 2))
    show("B dopo", run(brain, mb, orn_b, 2))

    mb.reset()
    brain.reset()
    print("\nCONTROLLO: 10 s di frutta da sola, senza rinforzo (non deve imparare niente)")
    fruit = np.concatenate([brain.find(g) for g in ODORS["fruity"]])
    fruit = (fruit, np.ones(len(fruit)))
    show("frutta all'inizio", run(brain, mb, fruit, 1))
    run(brain, mb, fruit, 9)
    run(brain, mb, none, 1)
    show("frutta dopo", run(brain, mb, fruit, 2))
    show("A dopo", run(brain, mb, orn_a, 2))
    print(f"  memoria cambiata: {100 * mb.strength():.2f}% dei pesi KC->MBON")

    tt = time.perf_counter()
    d = np.zeros(brain.N, np.int32)
    for _ in range(200):
        mb.update(d, WIN)
    print(f"\nCosto della memoria: {(time.perf_counter() - tt) / 200 * 1000:.2f} ms per finestra da 50 ms")
    print(f"\nFatto in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
