r"""
Touch: what happens in the brain when the fly gets hit
------------------------------------------------------
A hit in the game will become a stimulus on the mechanosensory neurons of the
head (bristles and antennae). Before hooking it up, we find out:

  1. which mechanosensory groups exist in the connectome
  2. for each one, stimulated ON THE LEFT ONLY (150 Hz, 0.5 s):
       - whether it drives the Giant Fiber (escape) and on which side
       - which descending neurons light up (the commands to the body:
         among them are the cleaning ones, i.e. "grooming")
       - how much the whole brain lights up

Uses the server configuration.

Usage:
  python scripts\14_touch.py
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
    print(f"Brain loaded in {time.perf_counter() - t0:.1f} s")

    cls = n.get("cell_class", pd.Series("", index=n.index)).fillna("").astype(str)
    sc = n["super_class"].fillna("").astype(str)
    ct = n["cell_type"].fillna("").astype(str)
    side = n["side"].fillna("").astype(str)
    mech = (cls.str.contains("mechano", case=False) |
            (sc.eq("sensory") & ct.str.match(r"^(JO|BM_|bristle)", case=False))).to_numpy()
    print(f"\nMechanosensory neurons: {mech.sum():,}")
    print("Classes: " + ", ".join(f"{k}({v})" for k, v in cls[mech].value_counts().head(10).items()))

    grp = ct.where(ct != "", "(no type) " + cls)
    groups = pd.Series(np.arange(len(n)))[mech].groupby(grp[mech].to_numpy()).apply(list)
    groups = groups[groups.apply(len) >= 2].sort_values(key=lambda s: s.apply(len), ascending=False)
    print(f"Groups to test: {len(groups)}")

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
            "group": name, "neurons": len(idx), "stimulated (L)": len(use),
            "GF L": int(c[circ.gf["L"]].sum()), "GF R": int(c[circ.gf["R"]].sum()),
            "spikes/s": int(c.sum() / s),
            "most active descending": ", ".join(f"{k} {v / s:.0f}Hz" for k, v in dn.items()) or "-",
        })

    df = pd.DataFrame(rows).sort_values(["GF L", "spikes/s"], ascending=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 90)
    print(f"\nEach group stimulated on the left at {RATE:.0f} Hz for {MS / 1000} s "
          f"({time.perf_counter() - t0:.0f} s in total):")
    print(df.to_string(index=False))
    print("\nWe're looking for: groups that drive the Giant Fiber (escape) and groups that drive")
    print("the cleaning descending neurons. Paste the output for me.")


if __name__ == "__main__":
    main()
