r"""
La miccia: come parte l'incendio, millisecondo per millisecondo
----------------------------------------------------------------
Accende un glomerulo che si incendia (default DM1) e:

  1. mostra chi spara in ognuno dei primi 30 ms (l'ordine di accensione)
  2. per i tipi più attivi: quanti neuroni, frequenza, neurotrasmettitore
     previsto dal connettoma (e quindi se nel modello eccitano o inibiscono)
  3. per i 3 tipi che bruciano di più: chi li accende (ingressi eccitatori
     e inibitori più forti, pesati per quanto hanno sparato)

Uso:
  python scripts\10_miccia.py
  python scripts\10_miccia.py --glom ORN_DM4
"""

import argparse

import numpy as np
import pandas as pd
import scipy.sparse as sp

from flysim import RECOMMENDED, FlyBrain

NT_SIGN = {"acetylcholine": "+", "gaba": "-", "glutamate": "-"}


def type_of(brain):
    return brain.neurons["cell_type"].fillna("?").astype(str).to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glom", default="ORN_DM1")
    ap.add_argument("--hz", type=float, default=20.0)
    args = ap.parse_args()

    brain = FlyBrain(dt=0.2, seed=0, params=RECOMMENDED)
    types = type_of(brain)
    nts = brain.neurons["top_nt"].fillna("?").astype(str).to_numpy()
    stim = brain.find(args.glom)
    print(f"Stimolo: {args.glom} ({len(stim)} ORN) a {args.hz:.0f} Hz\n")

    # 1. ordine di accensione ------------------------------------------
    brain.reset()
    brain.set_stimulus(stim, args.hz)
    print("=== 1. Chi spara nei primi 30 ms (esclusi gli ORN stimolati)")
    is_stim = np.zeros(brain.N, bool)
    is_stim[stim] = True
    for ms in range(30):
        c0 = brain.counts.copy()
        brain.run(1.0)
        d = brain.counts - c0
        d[is_stim] = 0
        if d.sum() == 0:
            continue
        s = pd.Series(d).groupby(types).sum()
        s = s[s > 0].sort_values(ascending=False).head(5)
        print(f"    {ms + 1:3d} ms  {int(d.sum()):6d} spike   " + ", ".join(f"{k} {v}" for k, v in s.items()))

    brain.run(70.0)                                   # fino a 100 ms
    counts_all = brain.counts.copy()                  # con gli ORN: servono per "chi li accende"
    counts = counts_all.copy()
    counts[is_stim] = 0

    # 2. identikit dei più attivi -------------------------------------
    print("\n=== 2. I più attivi nei primi 100 ms")
    df = pd.DataFrame({"type": types, "nt": nts, "s": counts})
    g = df.groupby("type").agg(n=("s", "size"), spikes=("s", "sum"),
                               nt=("nt", lambda x: x.value_counts().index[0]))
    g = g[g.spikes > 0].sort_values("spikes", ascending=False).head(12)
    print(f"    {'tipo':14s} {'neuroni':>7s} {'spike':>7s} {'Hz/neur.':>9s}  neurotrasm. (nel modello)")
    for t, r in g.iterrows():
        hz = r.spikes / r.n / 0.1
        sign = NT_SIGN.get(r.nt, "+")
        print(f"    {t:14s} {r.n:7d} {r.spikes:7d} {hz:9.0f}  {r.nt} ({'eccita' if sign == '+' else 'inibisce'})")

    # 3. chi accende i 3 tipi che bruciano di più ----------------------
    W = sp.csc_matrix((brain.data, brain.indices, brain.indptr), shape=(brain.N, brain.N)).tocsr()
    print("\n=== 3. Chi li accende (ingressi pesati per gli spike di chi li manda)")
    for t in g.index[:3]:
        rows = np.flatnonzero(types == t)
        contrib = np.asarray(W[rows].multiply(counts_all[None, :]).sum(axis=0)).ravel()  # per presinaptico
        s = pd.Series(contrib).groupby(types).sum()
        exc = s[s > 0].sort_values(ascending=False).head(5)
        inh = s[s < 0].sort_values().head(3)
        print(f"    {t}:")
        print("        eccitano: " + ", ".join(f"{k} {v:,.0f}" for k, v in exc.items()))
        print("        frenano:  " + (", ".join(f"{k} {v:,.0f}" for k, v in inh.items()) or "(nessuno attivo)"))

    print("\nSe tra chi eccita compaiono interneuroni locali (lLN...) segnati come acetilcolina,")
    print("abbiamo trovato l'amplificatore.")


if __name__ == "__main__":
    main()
