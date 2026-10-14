r"""
Map of olfactory runaway activity
---------------------------------
Activates ONE glomerulus at a time (both antennae, low intensity) with the
server configuration, and for each one answers three questions:

  1. does it run away?          spikes per second, spurious takeoffs
  2. what triggers it?          the most active neuron types at the moment it starts
  3. where does it spread?      most active classes and types over the whole runaway episode

It also checks the APL (the neuron that should keep the Kenyon cells quiet):
if it fires at its maximum it is saturated and cannot hold them back.

Usage:
  python scripts\09_runaway_map.py
  python scripts\09_runaway_map.py --hz 10 --sec 1.5
"""

import argparse

import numpy as np

import circuits
from circuits import FlyCircuits
from flysim import RECOMMENDED, FlyBrain

WINDOW_MS = 50.0
IGNITION = 1000          # spikes in a 50 ms window (= 20,000/s): runaway activity


def top_types(brain, counts, exclude, n=8):
    c = counts.copy()
    c[exclude] = 0
    df = brain.neurons.assign(s=c)
    t = df.groupby(df["cell_type"].fillna("?"))["s"].sum().sort_values(ascending=False)
    t = t[t > 0].head(n)
    return ", ".join(f"{k} {v:,}" for k, v in t.items()) or "(none)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hz", type=float, default=20.0, help="rate on the glomerulus ORNs")
    ap.add_argument("--sec", type=float, default=1.0, help="seconds per glomerulus")
    args = ap.parse_args()

    brain = FlyBrain(dt=0.2, seed=0, params=RECOMMENDED)
    circ = FlyCircuits(brain)
    gf = np.concatenate([circ.gf["L"], circ.gf["R"]])
    apl = brain.find("APL")
    print(f"Server configuration (interneurons fixed: {brain.ln_fixed}). APL found: {len(apl)}. "
          f"Stimulus: {args.hz:.0f} Hz for {args.sec} s on one glomerulus at a time.")

    glomeruli = [g for types in circuits.ODORS.values() for g in types]
    for g in glomeruli:
        idx = brain.find(g)
        if len(idx) == 0:
            print(f"\n=== {g}: not found")
            continue
        brain.reset()
        wins = []
        for _ in range(int(args.sec * 1000 / WINDOW_MS)):
            brain.set_stimulus(idx, args.hz)
            c0 = brain.counts.copy()
            brain.run(WINDOW_MS)
            wins.append(brain.counts - c0)

        per_win = np.array([w.sum() for w in wins])
        total = sum(wins)
        sps = per_win.sum() / args.sec
        gf_spikes = int(total[gf].sum())
        onset = next((i for i, s in enumerate(per_win) if s > IGNITION), None)
        status = f"YES, after {onset * WINDOW_MS:.0f} ms" if onset is not None else "no"
        print(f"\n=== {g}  ({len(idx)} ORN)   runaway: {status}   "
              f"{sps:,.0f} spikes/s   spurious takeoffs (GF spikes): {gf_spikes}")
        print("    spikes every 50 ms: " + " ".join(f"{int(s)}" for s in per_win[:12])
              + (" ..." if len(per_win) > 12 else ""))

        if onset is not None:
            start = max(0, onset - 1)
            lead = sum(wins[start:onset + 1])
            print(f"    triggered by:   {top_types(brain, lead, idx)}")
            print(f"    spreads to:     {top_types(brain, total, idx)}")
            df = brain.neurons.assign(s=total)
            cls = df.groupby(df["super_class"].fillna("?"))["s"].sum().sort_values(ascending=False).head(5)
            print("    by class:       " + ", ".join(f"{k} {v:,}" for k, v in cls.items()))
        else:
            print(f"    most active:    {top_types(brain, total, idx)}")

        if len(apl):
            print(f"    APL: {total[apl].sum() / len(apl) / args.sec:.0f} Hz "
                  f"(above ~350 Hz it is practically saturated)")

    print("\nWith this we choose where to intervene (graded APL, gain control, ...).")


if __name__ == "__main__":
    main()
