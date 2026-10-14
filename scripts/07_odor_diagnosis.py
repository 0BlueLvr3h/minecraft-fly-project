r"""
Diagnosis: why the odor triggers runaway activity in the brain, and whether adaptation stops it
----------------------------------------------------------------------------------------------
For each model variant:
  1. presents the fruit odor for 0.5 s, then removes it and watches for 1 s
     whether the brain calms down (spikes every 250 ms)
  2. checks that the escape still works (Giant Fiber with a running player)
  3. measures how strongly the odor-perception neurons light up

For the original model it also shows WHERE the activity runs away: most active
classes and types.

Usage:
  python scripts\07_odor_diagnosis.py
"""

import time

import numpy as np

from circuits import FlyCircuits, _rate
from flysim import FlyBrain

DT = 0.2
WINDOW_MS = 50.0
# (name, model parameters, escape exempt from adaptation?)
VARIANTS = [
    ("Shiu original",                        {}, True),
    ("strong adapt. on everything",          {"adapt_b": 5.0, "tau_a": 300.0}, False),
    ("strong adapt., escape exempt",         {"adapt_b": 5.0, "tau_a": 300.0}, True),
    ("medium adapt., escape exempt",         {"adapt_b": 2.0, "tau_a": 300.0}, True),
    ("strong adapt., escape exempt, no KC->KC", {"adapt_b": 5.0, "tau_a": 300.0,
                                                 "kc_recurrent": False}, True),
]
NONE = {"L": 0.0, "R": 0.0}


def smell_of(circ, name, level):
    s = {n: dict(NONE) for n in circ.odor}
    if name:
        s[name] = {"L": level, "R": level}
    return s


def run_windows(brain, circ, loom, smell, ms):
    """Simulate in windows like the server; return the spikes per window."""
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
            print(f"    (KC->KC connections cut: {brain.kc_cut:,})")

        # 1. odor on, then off
        brain.reset()
        w0 = time.perf_counter()
        on = run_windows(brain, circ, dict(NONE), smell_of(circ, "fruity", 0.3), 500)
        off = run_windows(brain, circ, dict(NONE), smell_of(circ, None, 0), 1000)
        wall = time.perf_counter() - w0
        per250 = [int(sum(d.sum() for d in (on + off)[i:i + 5])) for i in range(0, 30, 5)]
        print("    spikes every 250 ms (odor ON for the first 2, then OFF):")
        print("     " + "  ".join(f"{s:>9,d}" for s in per250))
        calm = "calms down" if per250[-1] < 0.05 * max(per250[:2] + [1]) else "does NOT calm down"
        print(f"    -> {calm}   (rt during the test: {1.5 / wall:.2f})")

        perc = np.mean([_rate(d, circ.perceive_idx["fruity"], WINDOW_MS / 1000) for d in on])
        print(f"    'fruity' perception during the odor: {perc:.0f} Hz")

        if label == "Shiu original":
            total = sum(on)
            df = brain.neurons.assign(spikes=total)
            by_class = df.groupby(df["super_class"].fillna("?"))["spikes"].sum()
            by_class = by_class.sort_values(ascending=False).head(6)
            print("    where it runs away (spikes per class): "
                  + ", ".join(f"{k} {v:,}" for k, v in by_class.items()))
            by_type = df.groupby(df["cell_type"].fillna("?"))["spikes"].sum()
            by_type = by_type.sort_values(ascending=False).head(8)
            print("    most active types: " + ", ".join(f"{k} {v:,}" for k, v in by_type.items()))

        # 2. escape
        brain.reset()
        esc = run_windows(brain, circ, {"L": 0.3, "R": 0.0}, smell_of(circ, None, 0), 1000)
        gf = sum(int(d[circ.gf["L"]].sum() + d[circ.gf["R"]].sum()) for d in esc)
        first = next((i for i, d in enumerate(esc)
                      if d[circ.gf["L"]].sum() + d[circ.gf["R"]].sum() > 0), None)
        when = f"first takeoff after {first * WINDOW_MS:.0f} ms" if first is not None else "NO takeoff"
        print(f"    escape: Giant Fiber {gf} spikes in 1 s, {when}")
        print(f"    ({time.perf_counter() - t0:.0f} s for this variant)")

    print("\nWe're looking for the variant that calms down, keeps up with real time and still escapes.")


if __name__ == "__main__":
    main()
