r"""
Il nucleo dell'incendio: chi resta acceso quando l'odore se ne va
------------------------------------------------------------------
Con la configurazione "fedele" (interneuroni corretti + taglio KC->KC) l'odore
accende il cervello e l'attività NON si spegne quando l'odore sparisce.

Lo script:
  1. dà la frutta per 1 s, poi la toglie e aspetta 1 s
  2. registra per 0,5 s chi continua a sparare DA SOLO (nessuno stimolo)
  3. per i tipi più attivi mostra: neurotrasmettitore previsto, frequenza,
     e quanta della loro eccitazione arriva DAL PROPRIO TIPO (auto-alimentazione)
  4. per i 5 più attivi: chi li alimenta

Configurazione: interneuroni corretti, taglio KC->KC, neuromodulatori lenti.

Uso:
  python scripts\13_nucleo.py
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp

from circuits import FlyCircuits
from flysim import FlyBrain

FEDELE = {"fix_ln_modulatory": True, "kc_recurrent": False, "modulators": "all"}
NONE = {"L": 0.0, "R": 0.0}


def main():
    brain = FlyBrain(dt=0.2, seed=0, params=FEDELE)
    circ = FlyCircuits(brain)
    types = brain.neurons["cell_type"].fillna("?").astype(str).to_numpy()
    nts = brain.neurons["top_nt"].fillna("?").astype(str).to_numpy()
    sc = brain.neurons["super_class"].fillna("?").astype(str).to_numpy()
    no_odor = {n: dict(NONE) for n in circ.odor}

    # 1. odore, poi niente
    smell = dict(no_odor)
    smell["fruity"] = {"L": 1.0, "R": 1.0}
    circ.apply(dict(NONE), smell)
    brain.run(1000)
    circ.apply(dict(NONE), no_odor)
    brain.run(1000)

    # 2. chi spara da solo
    c0 = brain.counts.copy()
    brain.run(500)
    counts = brain.counts - c0
    print(f"Senza nessuno stimolo da 1 s: {counts.sum() * 2:,} spike/s, "
          f"{(counts > 0).sum():,} neuroni attivi")
    cls = pd.Series(counts).groupby(sc).sum().sort_values(ascending=False).head(6)
    print("per classe: " + ", ".join(f"{k} {v * 2:,}/s" for k, v in cls.items()))

    # 3. identikit e auto-alimentazione
    W = sp.csc_matrix((brain.data, brain.indices, brain.indptr), shape=(brain.N, brain.N)).tocsr()
    df = pd.DataFrame({"type": types, "nt": nts, "s": counts})
    g = df.groupby("type").agg(n=("s", "size"), spikes=("s", "sum"),
                               nt=("nt", lambda x: x.value_counts().index[0]))
    g = g[g.spikes > 0].sort_values("spikes", ascending=False).head(15)
    print(f"\n    {'tipo':16s} {'neur.':>5s} {'Hz/neur.':>9s}  {'neurotrasm.':14s} {'da sé stesso':>12s}")
    feed = {}
    for t, r in g.iterrows():
        rows = np.flatnonzero(types == t)
        contrib = np.asarray(W[rows].multiply(counts[None, :]).sum(axis=0)).ravel()
        s = pd.Series(contrib).groupby(types).sum()
        exc = s[s > 0]
        self_share = exc.get(t, 0.0) / exc.sum() if exc.sum() > 0 else 0.0
        feed[t] = s
        print(f"    {t:16s} {r.n:5d} {r.spikes / r.n * 2:9.0f}  {r.nt:14s} {self_share * 100:11.0f}%")

    # 4. chi alimenta i primi 5
    print("\nChi alimenta i più attivi (eccitazione pesata per gli spike di chi la manda):")
    for t in list(g.index[:5]):
        s = feed[t]
        exc = s[s > 0].sort_values(ascending=False).head(5)
        inh = s[s < 0].sort_values().head(2)
        print(f"    {t}:")
        print("        eccitano: " + ", ".join(
            f"{k} [{pd.Series(nts[types == k]).value_counts().index[0]}] {v:,.0f}" for k, v in exc.items()))
        print("        frenano:  " + (", ".join(f"{k} {v:,.0f}" for k, v in inh.items()) or "(nessuno attivo)"))

    print("\nCerchiamo anelli tra neuroni segnati come modulatori (dopamina, serotonina,")
    print("octopamina) o tipi che si alimentano da soli.")


if __name__ == "__main__":
    main()
