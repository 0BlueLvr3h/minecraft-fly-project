r"""
Tatto: cosa succede nel cervello quando la mosca viene colpita
---------------------------------------------------------------
Un colpo in gioco diventerà uno stimolo sui neuroni meccanosensoriali della
testa (setole e antenne). Prima di collegarlo, scopriamo:

  1. quali gruppi meccanosensoriali esistono nel connettoma
  2. per ognuno, stimolato SOLO A SINISTRA (150 Hz, 0,5 s):
       - se accende il Giant Fiber (fuga) e da che lato
       - quali neuroni discendenti si accendono (i comandi verso il corpo:
         tra questi ci sono quelli della pulizia, il "grooming")
       - quanto si accende il cervello in tutto

Usa la configurazione del server.

Uso:
  python scripts\14_tatto.py
"""

import time

import numpy as np
import pandas as pd

from circuits import FlyCircuits
from flysim import RECOMMENDED, FlyBrain

RATE = 150.0
MS = 500.0


def main():
    t0 = time.perf_counter()
    brain = FlyBrain(dt=0.2, seed=0, params=RECOMMENDED)
    circ = FlyCircuits(brain)
    n = brain.neurons
    print(f"Cervello caricato in {time.perf_counter() - t0:.1f} s")

    cls = n.get("cell_class", pd.Series("", index=n.index)).fillna("").astype(str)
    sc = n["super_class"].fillna("").astype(str)
    ct = n["cell_type"].fillna("").astype(str)
    side = n["side"].fillna("").astype(str)
    mech = (cls.str.contains("mechano", case=False) |
            (sc.eq("sensory") & ct.str.match(r"^(JO|BM_|bristle)", case=False))).to_numpy()
    print(f"\nNeuroni meccanosensoriali: {mech.sum():,}")
    print("Classi: " + ", ".join(f"{k}({v})" for k, v in cls[mech].value_counts().head(10).items()))

    grp = ct.where(ct != "", "(senza tipo) " + cls)
    groups = pd.Series(np.arange(len(n)))[mech].groupby(grp[mech].to_numpy()).apply(list)
    groups = groups[groups.apply(len) >= 2].sort_values(key=lambda s: s.apply(len), ascending=False)
    print(f"Gruppi da provare: {len(groups)}")

    desc = (sc == "descending").to_numpy()
    rows = []
    for name, idx in groups.items():
        idx = np.asarray(idx)
        left = idx[side.to_numpy()[idx] == "left"]
        use = left if len(left) else idx
        brain.reset()
        brain.set_stimulus(use, RATE)
        brain.run(MS)
        c = brain.counts
        s = MS / 1000
        dn = pd.Series(c[desc], index=ct[desc].to_numpy()).groupby(level=0).sum()
        dn = dn[dn > 0].sort_values(ascending=False).head(4)
        rows.append({
            "gruppo": name, "neuroni": len(idx), "stimolati (sx)": len(use),
            "GF sx": int(c[circ.gf["L"]].sum()), "GF dx": int(c[circ.gf["R"]].sum()),
            "spike/s": int(c.sum() / s),
            "discendenti più attivi": ", ".join(f"{k} {v / s:.0f}Hz" for k, v in dn.items()) or "-",
        })

    df = pd.DataFrame(rows).sort_values(["GF sx", "spike/s"], ascending=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 90)
    print(f"\nOgni gruppo stimolato a sinistra a {RATE:.0f} Hz per {MS / 1000} s "
          f"({time.perf_counter() - t0:.0f} s in tutto):")
    print(df.to_string(index=False))
    print("\nCerchiamo: gruppi che accendono il Giant Fiber (fuga) e gruppi che accendono")
    print("discendenti della pulizia. Incollami l'output.")


if __name__ == "__main__":
    main()
