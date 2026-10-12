r"""
Corpo fungiforme: la mosca può imparare?
-----------------------------------------
Prima di aggiungere l'apprendimento controlliamo che il corpo fungiforme del
modello si comporti come quello vero:

  1. CODIFICA   un odore accende solo poche cellule di Kenyon (nella mosca vera
                circa il 5-10%), e odori diversi ne accendono gruppi diversi:
                è così che la mosca li distingue. Proviamo frutta, marcio e
                quattro "odori di player" (15 glomeruli a caso ciascuno).
  2. RINFORZO   lo zucchero accende i dopaminergici della ricompensa (PAM)?
                un colpo accende quelli della punizione (PPL1)?
  3. COMPARTIMENTI  per ogni dopaminergico, su quali neuroni di uscita (MBON)
                ha sinapsi: è la mappa che userà la regola di apprendimento.

Uso:
  python scripts\15_corpo_fungiforme.py
"""

import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

from circuits import ODOR_RATE, ODORS, TASTE_RATE, TOUCH_RATE, FlyCircuits
from flysim import PROC, RECOMMENDED, FlyBrain

MS = 500.0
SEED = 7
# Glomeruli che non usiamo per gli odori dei player: già usati (frutta, marcio) o con
# un significato innato forte (feromoni: DA1, VA1d, VA1v, DL3).
RESERVED = {g for gl in ODORS.values() for g in gl} | {"ORN_DA1", "ORN_VA1d", "ORN_VA1v", "ORN_DL3"}
PLAYER_GLOMERULI = 15         # un odore di un corpo accende molti glomeruli


def rates_by_type(brain, idx, ms):
    ct = brain.neurons["cell_type"].fillna("?").astype(str).to_numpy()
    s = pd.Series(brain.counts[idx] / (ms / 1000), index=ct[idx])
    return s.groupby(level=0).mean()


def main():
    t0 = time.perf_counter()
    brain = FlyBrain(dt=0.2, seed=SEED, params=RECOMMENDED)
    circ = FlyCircuits(brain)
    n = brain.neurons
    ct = n["cell_type"].fillna("").astype(str)
    print(f"Cervello caricato in {time.perf_counter() - t0:.1f} s ({brain.engine})")

    kc = brain.find_prefix("KC")
    mbon = brain.find_prefix("MBON")
    pam = brain.find_prefix("PAM")
    ppl1 = brain.find_prefix("PPL1")
    apl = brain.find("APL")
    pn = n["idx"].to_numpy()[n["cell_class"].fillna("").eq("ALPN").to_numpy()]
    print(f"KC {len(kc)}, MBON {len(mbon)}, PAM {len(pam)}, PPL1 {len(ppl1)}, APL {len(apl)}, PN {len(pn)}")

    # ------------------------------------------------------------ 1. codifica
    rng = np.random.default_rng(1)
    pool = sorted(t for t in ct[ct.str.startswith("ORN_")].unique() if t not in RESERVED)
    odors = {k: v for k, v in ODORS.items()}
    for i in range(4):
        odors[f"player{i + 1}"] = list(rng.choice(pool, PLAYER_GLOMERULI, replace=False))

    print(f"\n1. CODIFICA: ogni odore a {ODOR_RATE:.0f} Hz sugli ORN per {MS / 1000} s")
    sets, rows = {}, []
    for name, gl in odors.items():
        for conc in (1.0, 0.3):
            orn = np.concatenate([brain.find(g) for g in gl])
            brain.reset()
            brain.set_stimulus(orn, ODOR_RATE * conc)
            brain.run(MS)
            c = brain.counts
            act = kc[c[kc] >= 1]
            if conc == 1.0:
                sets[name] = set(act.tolist())
            rows.append({
                "odore": name, "conc": conc, "glomeruli": " ".join(g[4:] for g in gl),
                "PN Hz": round(c[pn].sum() / max(1, (c[pn] > 0).sum()) / (MS / 1000), 1),
                "KC attive %": round(100 * len(act) / len(kc), 1),
                "KC >=3 spike %": round(100 * (c[kc] >= 3).sum() / len(kc), 1),
                "APL Hz": round(c[apl].mean() / (MS / 1000), 1),
                "MBON attivi": int((c[mbon] > 0).sum()),
                "cervello spike/s": int(c.sum() / (MS / 1000)),
            })
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).to_string(index=False))

    names = list(sets)
    jac = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            u = len(sets[a] | sets[b])
            jac.loc[a, b] = round(len(sets[a] & sets[b]) / u, 2) if u else np.nan
    print("\nSovrapposizione delle KC accese (0 = gruppi diversi, 1 = identici):")
    print(jac.to_string())

    # ------------------------------------------------------------ 2. rinforzo
    print(f"\n2. RINFORZO: dopaminergici accesi da zucchero, colpo e odore ({MS / 1000} s)")
    stimuli = {
        "zucchero": (circ.sugar, TASTE_RATE),
        "colpo (setole+JO)": (np.concatenate([circ.touch["L"], circ.touch["R"]]), TOUCH_RATE),
        "frutta": (np.concatenate([circ.odor["fruity"]["L"], circ.odor["fruity"]["R"]]), ODOR_RATE),
        "niente": (np.array([], np.int64), 0.0),
    }
    cols = {}
    for name, (idx, hz) in stimuli.items():
        brain.reset()
        brain.set_stimulus(idx, hz)
        brain.run(MS)
        cols[name] = pd.concat([rates_by_type(brain, pam, MS), rates_by_type(brain, ppl1, MS)])
    dan = pd.DataFrame(cols).round(1)
    print(dan.to_string())

    # ------------------------------------------------------------ 3. compartimenti
    print("\n3. COMPARTIMENTI: per ogni tipo dopaminergico, i MBON su cui ha più sinapsi")
    W = sp.load_npz(PROC / "weights.npz").tocsr()           # [post x pre], prima delle correzioni
    ctn = ct.to_numpy()
    dan_all = np.concatenate([pam, ppl1])
    sub = abs(W[mbon][:, dan_all]).toarray()                 # MBON x DAN
    df = pd.DataFrame(sub, index=ctn[mbon], columns=ctn[dan_all])
    df = df.groupby(level=0).sum().T.groupby(level=0).sum()  # tipo DAN x tipo MBON
    for d, row in df.iterrows():
        top = row[row > 0].sort_values(ascending=False).head(3)
        print(f"  {d:8s} -> " + (", ".join(f"{k} ({int(v)})" for k, v in top.items()) or "nessun MBON"))
    kcm = abs(W[mbon][:, kc]).sum()
    print(f"\nSinapsi KC -> MBON in tutto: {int(kcm):,}")
    print(f"\nFatto in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
