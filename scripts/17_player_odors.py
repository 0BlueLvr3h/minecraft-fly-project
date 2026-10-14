r"""
Player odors: which ones can the brain tell apart?
--------------------------------------------------
The fly recognizes a player only by its odor, and learns it in the Kenyon cells (KCs).
For each player to stay "themselves", their odor must:
  - activate enough KCs (otherwise the memory has little to write on)
  - always activate the same ones (otherwise it seems a new odor every time)
  - overlap little with the other players' odors (otherwise it confuses them)
  - NOT activate the Giant Fiber: for a fly the odor of a body is not an innate
    danger (some glomerulus mixes reach the escape circuit and would make it flee on their own)
  - cost the brain little: with several players nearby the odors add up, and an odor
    that activates half the brain slows the simulation below real time

First we test the glomeruli one by one and discard the "inefficient" ones: those that
strongly activate the rest of the brain and the mushroom body only weakly. Then we generate many candidate odors (a group of glomeruli, each with its own intensity),
let the brain smell them twice at about 2 blocks away and keep the
best ones. The brain itself is not touched: we only choose the odors.

Writes data/processed/player_odors.json, which the server uses to give each
player an odor. Only needs to be redone if the connectome or the brain configuration changes.

Usage:
  python scripts\17_player_odors.py              (tests the odors and chooses: a few minutes)
  python scripts\17_player_odors.py --reselect   (chooses again among those already tested: seconds)
"""

import json
import sys
import time

import numpy as np

from circuits import ODOR_RATE, ODORS, FlyCircuits, player_rate
from flysim import PROC, RECOMMENDED, FlyBrain

CANDIDATES = 300
KEEP = 16                 # how many distinct odors (beyond that: two players may share one)
CONC = 0.6                # about a player 2 blocks away
MS = 1000.0
MAX_COST = 40000          # whole-brain spikes/s for an odor 2 blocks away (above: too heavy)
MAX_OVERLAP = 0.12        # maximum overlap between the KCs of two chosen odors
MIN_KC = 40               # minimum reliable KCs for an odor to count
# Excluded glomeruli: already used for fruit and rotten, or with a strong innate meaning:
# pheromones (DA1, VA1d, VA1v, DL3) and ammonia/amines (VM1, VC5, VM6m, VM6v, VM6l), which in the
# connectome reach the Giant Fiber: on their own they make the fly flee.
RESERVED = ({g for gl in ODORS.values() for g in gl}
            | {"ORN_DA1", "ORN_VA1d", "ORN_VA1v", "ORN_DL3"}
            | {"ORN_VM1", "ORN_VC5", "ORN_VM6m", "ORN_VM6v", "ORN_VM6l"})
# Inefficient glomerulus: on its own it costs more than GLOM_HEAVY spikes/s, or more than GLOM_COST
# while activating fewer than GLOM_MIN_KC Kenyon cells.
GLOM_HEAVY = 30000
GLOM_COST = 10000
GLOM_MIN_KC = 40
OUT = PROC / "player_odors.json"
CACHE = PROC / "player_odor_candidates.json"      # odors already tested, for --reselect


def simulate():
    """Lets the brain smell the glomeruli and then the candidate odors; saves everything to CACHE."""
    t0 = time.perf_counter()
    brain = FlyBrain(dt=0.2, seed=17, params=RECOMMENDED)
    circ = FlyCircuits(brain)
    gf = np.concatenate([circ.gf["L"], circ.gf["R"]])
    kc = brain.find_prefix("KC")
    ct = brain.neurons["cell_type"].fillna("").astype(str)
    pool = sorted(t for t in ct[ct.str.startswith("ORN_")].unique() if t not in RESERVED)
    print(f"Brain ready in {time.perf_counter() - t0:.1f} s; testing {len(pool)} glomeruli one by one")
    keep = []
    for g in pool:
        cost, n_kc = 0.0, 0
        for _ in range(2):                       # two trials: the brain is noisy
            brain.reset()
            brain.set_stimulus(brain.find(g), player_rate(CONC))
            brain.run(MS)
            cost += brain.counts.sum() / (MS / 1000) / 2
            n_kc += int((brain.counts[kc] >= 2).sum()) / 2
        bad = cost > GLOM_HEAVY or (cost > GLOM_COST and n_kc < GLOM_MIN_KC)
        if bad:
            print(f"  discard {g[4:]:5s}: {cost / 1000:.0f}k spikes/s for {n_kc:.0f} KC")
        else:
            keep.append(g)
    pool = keep
    print(f"{len(pool)} usable glomeruli, {CANDIDATES} odors to test")

    rng = np.random.default_rng(2024)
    cands = []
    for c in range(CANDIDATES):
        n = int(rng.integers(10, 19))
        gl = list(rng.choice(pool, n, replace=False))
        prof = [round(float(x), 2) for x in rng.uniform(0.3, 1.0, n)]
        idx = np.concatenate([brain.find(g) for g in gl])
        hz = np.concatenate([np.full(len(brain.find(g)), w) for g, w in zip(gl, prof)])
        sets, gf_spk, cost = [], 0, 0.0
        for _ in range(2):
            brain.reset()
            brain.set_stimulus(idx, player_rate(CONC) * hz)
            brain.run(MS)
            sets.append(set(kc[brain.counts[kc] >= 2].tolist()))
            gf_spk += int(brain.counts[gf].sum())
            cost += brain.counts.sum() / (MS / 1000) / 2
        reliable = sets[0] & sets[1]
        union = sets[0] | sets[1]
        cands.append({"glomeruli": gl, "profile": prof, "kc": sorted(reliable), "gf": gf_spk, "cost": cost,
                      "reliability": len(reliable) / max(1, len(union))})
        if (c + 1) % 20 == 0:
            print(f"  {c + 1}/{CANDIDATES} ({time.perf_counter() - t0:.0f} s)")
    CACHE.write_text(json.dumps({"n_kc": int(len(kc)), "candidates": cands}), encoding="utf-8")
    return cands, len(kc)


def select(cands, n_kc):
    """Chooses the largest group of odors that overlap little with each other."""
    for c in cands:
        c["kc"] = set(c["kc"])
    n_gf = sum(c["gf"] > 0 for c in cands)
    n_cost = sum(c["cost"] > MAX_COST for c in cands)
    good = [c for c in cands if len(c["kc"]) >= MIN_KC and c["gf"] == 0 and c["cost"] <= MAX_COST]
    print(f"\nDiscarded: {n_gf} activate the Giant Fiber, {n_cost} too heavy (> {MAX_COST} spikes/s)")
    print(f"Good odors (at least {MIN_KC} reliable KCs, no escape, light): {len(good)} of {len(cands)}")

    def overlap(a, b):
        return len(a["kc"] & b["kc"]) / len(a["kc"] | b["kc"])

    # conflicts: pairs of odors that overlap too much (the fly would confuse them)
    conflicts = [{j for j, b in enumerate(good) if j != i and overlap(a, b) > MAX_OVERLAP}
                 for i, a in enumerate(good)]
    score = [len(c["kc"]) * c["reliability"] for c in good]
    left, picked = set(range(len(good))), []
    while left and len(picked) < KEEP:
        # first the odor that excludes the fewest others; on a tie, the one the fly recognizes best
        i = min(left, key=lambda k: (len(conflicts[k] & left), -score[k]))
        picked.append(i)
        left -= conflicts[i] | {i}
    chosen = [good[i] for i in picked]
    # order: the first players get the most mutually different odors (the most similar pairs
    # come into play only when there are many players)
    if chosen:
        order = [max(chosen, key=lambda c: len(c["kc"]) * c["reliability"])]
        rest = [c for c in chosen if c is not order[0]]
        while rest:
            nxt = min(rest, key=lambda c: max(overlap(c, d) for d in order))
            order.append(nxt)
            rest.remove(nxt)
        chosen = order
    for c in chosen:
        c["overlap"] = max((overlap(c, d) for d in chosen if d is not c), default=0.0)

    print(f"Chose {len(chosen)} odors:")
    for i, c in enumerate(chosen):
        print(f"  {i:2d}: {len(c['glomeruli']):2d} glomeruli, reliable KCs {len(c['kc']):4d} "
              f"({100 * len(c['kc']) / n_kc:.1f}%), repeatability {c['reliability']:.2f}, "
              f"max overlap {c['overlap']:.2f}, cost {c['cost'] / 1000:.0f}k spikes/s")
    if len(chosen) < KEEP:
        print(f"Note: only {len(chosen)} sufficiently distinct odors, additional players will share them")
    return chosen


def main():
    t0 = time.perf_counter()
    if "--reselect" in sys.argv:
        if not CACHE.exists():
            raise SystemExit("No odors tested yet: run the script without --reselect")
        data = json.loads(CACHE.read_text(encoding="utf-8"))
        cands, n_kc = data["candidates"], data["n_kc"]
        print(f"Reselecting among {len(cands)} odors already tested")
    else:
        cands, n_kc = simulate()
    chosen = select(cands, n_kc)

    OUT.write_text(json.dumps([{"glomeruli": c["glomeruli"], "profile": c["profile"]} for c in chosen],
                              indent=1), encoding="utf-8")
    assign = PROC / "player_assign.json"          # the odors have changed: they are reassigned from scratch
    if assign.exists():
        assign.unlink()
        print("Player assignments reset (each one will get one of the new odors)")
    print(f"\nSaved {OUT} in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
