r"""
The core of the runaway activity: who stays active once the odor is gone
------------------------------------------------------------------------
With the "faithful" configuration (interneuron fix + KC->KC cut) the odor
excites the brain and the activity does NOT die out when the odor disappears.

The script:
  1. presents fruit for 1 s, then removes it and waits 1 s
  2. records for 0.5 s who keeps firing ON ITS OWN (no stimulus)
  3. for the most active types shows: predicted neurotransmitter, rate,
     and how much of their excitation comes FROM THEIR OWN TYPE (self-excitation)
  4. for the 5 most active: who feeds them

Configuration: interneuron fix, KC->KC cut, slow neuromodulators.

Usage:
  python scripts\13_runaway_core.py
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp

from circuits import FlyCircuits
from flysim import FlyBrain

FAITHFUL = {"fix_ln_modulatory": True, "kc_recurrent": False, "modulators": "all"}
NONE = {"L": 0.0, "R": 0.0}


def main():
    brain = FlyBrain(dt=0.2, seed=0, params=FAITHFUL)
    circ = FlyCircuits(brain)
    types = brain.neurons["cell_type"].fillna("?").astype(str).to_numpy()
    nts = brain.neurons["top_nt"].fillna("?").astype(str).to_numpy()
    sc = brain.neurons["super_class"].fillna("?").astype(str).to_numpy()
    no_odor = {n: dict(NONE) for n in circ.odor}

    # 1. odor, then nothing
    smell = dict(no_odor)
    smell["fruity"] = {"L": 1.0, "R": 1.0}
    circ.apply(dict(NONE), smell)
    brain.run(1000)
    circ.apply(dict(NONE), no_odor)
    brain.run(1000)

    # 2. who fires on its own
    c0 = brain.counts.copy()
    brain.run(500)
    counts = brain.counts - c0
    print(f"With no stimulus for 1 s: {counts.sum() * 2:,} spikes/s, "
          f"{(counts > 0).sum():,} active neurons")
    cls = pd.Series(counts).groupby(sc).sum().sort_values(ascending=False).head(6)
    print("by class: " + ", ".join(f"{k} {v * 2:,}/s" for k, v in cls.items()))

    # 3. profile and self-excitation
    W = sp.csc_matrix((brain.data, brain.indices, brain.indptr), shape=(brain.N, brain.N)).tocsr()
    df = pd.DataFrame({"type": types, "nt": nts, "s": counts})
    g = df.groupby("type").agg(n=("s", "size"), spikes=("s", "sum"),
                               nt=("nt", lambda x: x.value_counts().index[0]))
    g = g[g.spikes > 0].sort_values("spikes", ascending=False).head(15)
    print(f"\n    {'type':16s} {'neur.':>5s} {'Hz/neur.':>9s}  {'neurotransm.':14s} {'from itself':>12s}")
    feed = {}
    for t, r in g.iterrows():
        rows = np.flatnonzero(types == t)
        contrib = np.asarray(W[rows].multiply(counts[None, :]).sum(axis=0)).ravel()
        s = pd.Series(contrib).groupby(types).sum()
        exc = s[s > 0]
        self_share = exc.get(t, 0.0) / exc.sum() if exc.sum() > 0 else 0.0
        feed[t] = s
        print(f"    {t:16s} {r.n:5d} {r.spikes / r.n * 2:9.0f}  {r.nt:14s} {self_share * 100:11.0f}%")

    # 4. who feeds the top 5
    print("\nWho feeds the most active (excitation weighted by the spikes of the senders):")
    for t in list(g.index[:5]):
        s = feed[t]
        exc = s[s > 0].sort_values(ascending=False).head(5)
        inh = s[s < 0].sort_values().head(2)
        print(f"    {t}:")
        print("        excite:   " + ", ".join(
            f"{k} [{pd.Series(nts[types == k]).value_counts().index[0]}] {v:,.0f}" for k, v in exc.items()))
        print("        inhibit:  " + (", ".join(f"{k} {v:,.0f}" for k, v in inh.items()) or "(none active)"))

    print("\nWe're looking for loops among neurons labeled as modulatory (dopamine, serotonin,")
    print("octopamine) or types that feed themselves.")


if __name__ == "__main__":
    main()
