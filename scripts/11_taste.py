r"""
Taste: which gustatory neurons make the fly "eat"
-------------------------------------------------
In the Shiu et al. paper the model was validated precisely here: the sugar
receptors (GRNs) activate MN9, the motor neuron that extends the proboscis.

The script:
  1. finds the gustatory neurons and MN9 in the connectome
  2. stimulates each gustatory neuron type on its own (100 Hz, 1 s)
  3. measures how strongly MN9 lights up: the types that drive it the most
     are the "sugar" we will use in the game

It does this with the server configuration and with Shiu's original model,
to check that our fixes have not broken the circuit.

Usage:
  python scripts\11_taste.py
"""

import time

import numpy as np
import pandas as pd

from flysim import RECOMMENDED, FlyBrain

RATE = 100.0
MS = 1000.0

# FlyWire IDs used by Shiu et al. in the example notebook of their model
# (github.com/philshiu/Drosophila_brain_model, example.ipynb)
SHIU_SUGAR = [
    720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345,
    720575940617000768, 720575940630797113, 720575940632889389, 720575940621754367,
    720575940621502051, 720575940640649691, 720575940639332736, 720575940616885538,
    720575940639198653, 720575940620900446, 720575940617937543, 720575940632425919,
    720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
    720575940611875570,
]
SHIU_MN9 = [720575940660219265]


def by_root(brain, ids):
    m = brain.neurons["root_id"].isin(ids).to_numpy()
    return brain.neurons["idx"].to_numpy()[m]


def describe_group(brain, idx):
    sub = brain.neurons.iloc[idx]
    parts = []
    for col in ("cell_type", "hemibrain_type", "cell_class", "side"):
        if col in sub:
            vc = sub[col].fillna("-").astype(str).value_counts()
            parts.append(f"{col}: " + ", ".join(f"{k}({v})" for k, v in vc.head(4).items()))
    return " | ".join(parts)


def gustatory_groups(brain):
    n = brain.neurons
    cls = n.get("cell_class", pd.Series("", index=n.index)).fillna("").astype(str)
    ct = n["cell_type"].fillna("").astype(str)
    mask = cls.str.contains("gust", case=False) | ct.str.contains(r"GRN|^Gr\d", case=True)
    sub = n[mask.to_numpy()].copy()
    sub["grp"] = sub["cell_type"].fillna("").astype(str)
    sub.loc[sub["grp"] == "", "grp"] = "(no type) " + cls[mask.to_numpy()][sub["grp"] == ""]
    return sub, cls


def main():
    t0 = time.perf_counter()
    configs = [("server", FlyBrain(dt=0.2, seed=0, params=RECOMMENDED)),
               ("Shiu orig.", FlyBrain(dt=0.2, seed=0))]
    brain = configs[0][1]
    print(f"Brains loaded in {time.perf_counter() - t0:.1f} s")

    sub, cls = gustatory_groups(brain)
    sensory_cls = cls[brain.neurons["super_class"].fillna("").eq("sensory").to_numpy()]
    print("\nSensory neuron classes: "
          + ", ".join(f"{k}({v})" for k, v in sensory_cls.value_counts().head(12).items()))

    sugar = by_root(brain, SHIU_SUGAR)
    mn9 = by_root(brain, SHIU_MN9)
    print(f"\nShiu's sugar: {len(sugar)}/{len(SHIU_SUGAR)} found -> {describe_group(brain, sugar)}")
    print(f"Shiu's MN9:   {len(mn9)}/{len(SHIU_MN9)} found -> {describe_group(brain, mn9)}")
    # MN9's "siblings" (same type, other side)
    if len(mn9):
        r = brain.neurons.iloc[mn9[0]]
        for col in ("cell_type", "hemibrain_type"):
            v = r.get(col)
            if isinstance(v, str) and v:
                sib = brain.neurons[brain.neurons[col] == v]["idx"].to_numpy()
                if len(mn9) < len(sib) <= 4:              # only if it is a small type (MN9 L/R)
                    print(f"   MN9 extended with {col} = {v}: {len(sib)} neurons")
                    mn9 = sib
                break
    print(f"Gustatory neurons: {len(sub)} in {sub['grp'].nunique()} groups")
    if len(mn9) == 0 or len(sub) == 0:
        print("I need the right names: paste this output for me.")
        mt = brain.types_with_prefix("MN")
        print("Types starting with MN: " + ", ".join(mt.index[:40]))
        return

    groups = sub.groupby("grp")["idx"].apply(lambda s: s.to_numpy())
    if len(sugar):
        groups = pd.concat([pd.Series({"** Shiu's sugar **": sugar}), groups])
    rows = []
    for name, idx in groups.items():
        row = {"group": name, "neurons": len(idx)}
        for label, b in configs:
            b.reset()
            b.set_stimulus(idx, RATE)
            b.run(MS)
            row[f"MN9 Hz ({label})"] = b.counts[mn9].sum() / len(mn9) / (MS / 1000)
            row[f"spikes/s ({label})"] = int(b.counts.sum() / (MS / 1000))
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("MN9 Hz (server)", ascending=False)
    pd.set_option("display.width", 200)
    print(f"\nEach gustatory group alone at {RATE:.0f} Hz for {MS / 1000:.0f} s:")
    print(df.head(25).to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    print("\nThe groups at the top (high MN9) are the 'sugar'. Paste the output for me.")


if __name__ == "__main__":
    main()
