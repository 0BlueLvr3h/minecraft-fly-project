r"""
Conditioning: does the fly learn who you are?
---------------------------------------------
The classic Drosophila experiment, run on the connectome with memory.py, with the
same 50 ms windows as the game and with the player odors chosen by
17_player_odors.py (run it first).

Three players:
  A  feeds it: odor A together with the reward (PAM dopaminergic neurons)
  B  hits it: odor B with three hits (PPL1 dopaminergic neurons)
  C  a stranger: has never done anything to it

Then it smells A, B and C alone. Learned valence: 0 = neutral, > 0 = good (it
approaches), < 0 = bad (it avoids). We expect A positive, B negative, C close
to 0. The experiment is repeated several times: the brain is noisy like the real one.
At the end a control: fruit alone, without reinforcement, must teach nothing.

Usage:
  python scripts\16_conditioning.py
"""

import json
import time

import numpy as np

from circuits import ODOR_RATE, ODORS, FlyCircuits, player_rate
from flysim import PROC, RECOMMENDED, FlyBrain
from memory import MushroomBody

WIN = 50.0
NEAR, FAR = 0.6, 0.3      # concentration: about 2 and 4 blocks
REPEATS = 3


def odor(brain, glomeruli, profile):
    idx = [brain.find(g) for g in glomeruli]
    hz = [np.full(len(i), w) for i, w in zip(idx, profile)]
    return np.concatenate(idx), np.concatenate(hz)


def run(brain, mb, orn, seconds, conc=NEAR, reward=0.0, punish=0.0, fixed_hz=None):
    """Simulates `seconds` seconds in 50 ms windows; returns the mean valence of the last second."""
    vals = []
    n = int(seconds * 1000 / WIN)
    for k in range(n):
        if orn is None:
            brain.clear_stimulus()
        else:
            brain.set_stimulus(orn[0], (fixed_hz or player_rate(conc)) * orn[1])
        for idx, hz in mb.reinforcement(reward, punish):
            brain.add_stimulus(idx, hz)
        before = brain.counts.copy()
        brain.run(WIN)
        mb.update(brain.counts - before, WIN)
        if k >= n - int(1000 / WIN):
            vals.append(mb.valence)
    return float(np.mean(vals))


def main():
    t0 = time.perf_counter()
    f = PROC / "player_odors.json"
    if not f.exists():
        raise SystemExit("data/processed/player_odors.json is missing: run scripts\\17_player_odors.py first")
    odors = json.loads(f.read_text(encoding="utf-8"))
    brain = FlyBrain(dt=0.2, seed=3, params=RECOMMENDED)
    FlyCircuits(brain)
    mb = MushroomBody(brain)
    a, b, c = (odor(brain, o["glomeruli"], o["profile"]) for o in odors[:3])
    none = None
    print(f"Ready in {time.perf_counter() - t0:.1f} s: {len(mb.pos):,} KC->MBON synapses, "
          f"MBONs read: {(mb.comp == 1).sum()} PAM, {(mb.comp == -1).sum()} PPL1")

    rows = []
    for rep in range(REPEATS):
        mb.reset()
        brain.reset()
        r = {"A before": run(brain, mb, a, 1.5), "B before": run(brain, mb, b, 1.5)}
        run(brain, mb, a, 6, reward=1.0)                      # A feeds it
        run(brain, mb, none, 1)
        for _ in range(3):                                    # B hits it three times
            run(brain, mb, b, 0.5)
            run(brain, mb, b, 1.0, punish=1.0)
        run(brain, mb, none, 1)
        r.update({"A": run(brain, mb, a, 2), "B": run(brain, mb, b, 2), "C stranger": run(brain, mb, c, 2),
                  "A far": run(brain, mb, a, 2, conc=FAR), "B far": run(brain, mb, b, 2, conc=FAR)})
        rows.append(r)
        print(f"  trial {rep + 1}: " + "  ".join(f"{k} {v:+.2f}" for k, v in r.items()))

    print("\nMean over trials (valence: + good, - bad):")
    for k in rows[0]:
        v = [r[k] for r in rows]
        print(f"  {k:12s} {np.mean(v):+.2f}   (from {min(v):+.2f} to {max(v):+.2f})")

    mb.reset()
    brain.reset()
    fruit = np.concatenate([brain.find(g) for g in ODORS["fruity"]])
    fruit = (fruit, np.ones(len(fruit)))
    run(brain, mb, fruit, 10, fixed_hz=ODOR_RATE)
    run(brain, mb, none, 1)
    print(f"\nControl, 10 s of fruit without reinforcement: fruit {run(brain, mb, fruit, 2, fixed_hz=ODOR_RATE):+.2f}, "
          f"A {run(brain, mb, a, 2):+.2f}  (must stay at 0)")
    print(f"\nDone in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
