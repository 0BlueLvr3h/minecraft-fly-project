r"""
Diagnosi: perché l'odore "incendia" il cervello, e se l'adattamento lo spegne
-----------------------------------------------------------------------------
Per ogni variante del modello:
  1. dà l'odore di frutta per 0,5 s, poi lo toglie e guarda per 1 s se il
     cervello si calma (spike ogni 250 ms)
  2. controlla che la fuga funzioni ancora (Giant Fiber con un player che corre)
  3. misura quanto si accendono i neuroni della percezione dell'odore

Per il modello originale mostra anche DOVE brucia: classi e tipi più attivi.

Uso:
  python scripts\07_diagnosi_odore.py
"""

import time

import numpy as np

from circuits import FlyCircuits, _rate
from flysim import FlyBrain

DT = 0.2
WINDOW_MS = 50.0
# (nome, parametri del modello, fuga esente dall'adattamento?)
VARIANTS = [
    ("Shiu originale",                       {}, True),
    ("adatt. forte su tutto",                {"adapt_b": 5.0, "tau_a": 300.0}, False),
    ("adatt. forte, fuga esente",            {"adapt_b": 5.0, "tau_a": 300.0}, True),
    ("adatt. medio, fuga esente",            {"adapt_b": 2.0, "tau_a": 300.0}, True),
    ("adatt. forte, fuga esente, no KC->KC", {"adapt_b": 5.0, "tau_a": 300.0,
                                              "kc_recurrent": False}, True),
]
NONE = {"L": 0.0, "R": 0.0}


def smell_of(circ, name, level):
    s = {n: dict(NONE) for n in circ.odor}
    if name:
        s[name] = {"L": level, "R": level}
    return s


def run_windows(brain, circ, loom, smell, ms):
    """Simula a finestre come il server; restituisce gli spike per finestra."""
    out = []
    for _ in range(int(ms / WINDOW_MS)):
        circ.apply(loom, smell)
        c0 = brain.counts.copy()
        brain.run(WINDOW_MS)
        out.append(brain.counts - c0)
    return out


def main():
    for label, params, protect in VARIANTS:
        t0 = time.perf_counter()
        brain = FlyBrain(dt=DT, seed=0, params=params)
        circ = FlyCircuits(brain, protect_escape=protect)
        print(f"\n=== {label}  {params}")
        if brain.kc_cut:
            print(f"    (connessioni KC->KC tagliate: {brain.kc_cut:,})")

        # 1. odore acceso poi spento
        brain.reset()
        w0 = time.perf_counter()
        on = run_windows(brain, circ, dict(NONE), smell_of(circ, "fruity", 0.3), 500)
        off = run_windows(brain, circ, dict(NONE), smell_of(circ, None, 0), 1000)
        wall = time.perf_counter() - w0
        per250 = [int(sum(d.sum() for d in (on + off)[i:i + 5])) for i in range(0, 30, 5)]
        print("    spike ogni 250 ms (odore ON per i primi 2, poi OFF):")
        print("     " + "  ".join(f"{s:>9,d}" for s in per250))
        calm = "si calma" if per250[-1] < 0.05 * max(per250[:2] + [1]) else "NON si calma"
        print(f"    -> {calm}   (rt durante il test: {1.5 / wall:.2f})")

        perc = np.mean([_rate(d, circ.perceive_idx["fruity"], WINDOW_MS / 1000) for d in on])
        print(f"    percezione 'fruity' durante l'odore: {perc:.0f} Hz")

        if label == "Shiu originale":
            total = sum(on)
            df = brain.neurons.assign(spikes=total)
            by_class = df.groupby(df["super_class"].fillna("?"))["spikes"].sum()
            by_class = by_class.sort_values(ascending=False).head(6)
            print("    dove brucia (spike per classe): "
                  + ", ".join(f"{k} {v:,}" for k, v in by_class.items()))
            by_type = df.groupby(df["cell_type"].fillna("?"))["spikes"].sum()
            by_type = by_type.sort_values(ascending=False).head(8)
            print("    tipi più attivi: " + ", ".join(f"{k} {v:,}" for k, v in by_type.items()))

        # 2. fuga
        brain.reset()
        esc = run_windows(brain, circ, {"L": 0.3, "R": 0.0}, smell_of(circ, None, 0), 1000)
        gf = sum(int(d[circ.gf["L"]].sum() + d[circ.gf["R"]].sum()) for d in esc)
        first = next((i for i, d in enumerate(esc)
                      if d[circ.gf["L"]].sum() + d[circ.gf["R"]].sum() > 0), None)
        when = f"primo decollo dopo {first * WINDOW_MS:.0f} ms" if first is not None else "NESSUN decollo"
        print(f"    fuga: Giant Fiber {gf} spike in 1 s, {when}")
        print(f"    ({time.perf_counter() - t0:.0f} s per questa variante)")

    print("\nCerchiamo la variante che si calma, regge il tempo reale e scappa ancora.")


if __name__ == "__main__":
    main()
