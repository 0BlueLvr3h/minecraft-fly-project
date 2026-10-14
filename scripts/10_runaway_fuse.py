r"""
The fuse: how runaway activity ignites, millisecond by millisecond
------------------------------------------------------------------
Activates a glomerulus that triggers runaway activity (default DM1) and:

  1. shows who fires in each of the first 30 ms (the ignition order)
  2. for the most active types: how many neurons, rate, neurotransmitter
     predicted by the connectome (and therefore whether they excite or inhibit
     in the model)
  3. for the 3 types that fire the most: what drives them (strongest excitatory
     and inhibitory inputs, weighted by how much they fired)

Usage:
  python scripts\10_runaway_fuse.py
  python scripts\10_runaway_fuse.py --glom ORN_DM4
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
    print(f"Stimulus: {args.glom} ({len(stim)} ORN) at {args.hz:.0f} Hz\n")

    # 1. ignition order --------------------------------------------------
    brain.reset()
    brain.set_stimulus(stim, args.hz)
    print("=== 1. Who fires in the first 30 ms (stimulated ORNs excluded)")
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
        print(f"    {ms + 1:3d} ms  {int(d.sum()):6d} spikes  " + ", ".join(f"{k} {v}" for k, v in s.items()))

    brain.run(70.0)                                   # up to 100 ms
    counts_all = brain.counts.copy()                  # including the ORNs: needed for "what drives them"
    counts = counts_all.copy()
    counts[is_stim] = 0

    # 2. profile of the most active --------------------------------------
    print("\n=== 2. The most active in the first 100 ms")
    df = pd.DataFrame({"type": types, "nt": nts, "s": counts})
    g = df.groupby("type").agg(n=("s", "size"), spikes=("s", "sum"),
                               nt=("nt", lambda x: x.value_counts().index[0]))
    g = g[g.spikes > 0].sort_values("spikes", ascending=False).head(12)
    print(f"    {'type':14s} {'neurons':>7s} {'spikes':>7s} {'Hz/neur.':>9s}  neurotransm. (in the model)")
    for t, r in g.iterrows():
        hz = r.spikes / r.n / 0.1
        sign = NT_SIGN.get(r.nt, "+")
        print(f"    {t:14s} {r.n:7d} {r.spikes:7d} {hz:9.0f}  {r.nt} ({'excites' if sign == '+' else 'inhibits'})")

    # 3. what drives the 3 types that fire the most ----------------------
    W = sp.csc_matrix((brain.data, brain.indices, brain.indptr), shape=(brain.N, brain.N)).tocsr()
    print("\n=== 3. What drives them (inputs weighted by the spikes of the senders)")
    for t in g.index[:3]:
        rows = np.flatnonzero(types == t)
        contrib = np.asarray(W[rows].multiply(counts_all[None, :]).sum(axis=0)).ravel()  # per presynaptic neuron
        s = pd.Series(contrib).groupby(types).sum()
        exc = s[s > 0].sort_values(ascending=False).head(5)
        inh = s[s < 0].sort_values().head(3)
        print(f"    {t}:")
        print("        excite:   " + ", ".join(f"{k} {v:,.0f}" for k, v in exc.items()))
        print("        inhibit:  " + (", ".join(f"{k} {v:,.0f}" for k, v in inh.items()) or "(none active)"))

    print("\nIf local interneurons (lLN...) labeled as acetylcholine show up among the exciters,")
    print("we have found the amplifier.")


if __name__ == "__main__":
    main()
