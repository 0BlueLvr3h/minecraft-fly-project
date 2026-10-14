r"""
circuits - how the fly senses and how it moves
----------------------------------------------
This is the whole "translation" between the Minecraft world and the neurons:

  SENSES  (world -> neurons)
    eyes:      approaching players       -> LC4, LPLC2      (looming)
    antennae:  item odors                -> ORNs of the olfactory glomeruli
               each player's odor        -> its own mix of glomeruli (PlayerOdors)
    noise:     spontaneous activity      -> sensory neurons

  MOTORS  (neurons -> world)
    escape:    Giant Fiber (DNp01) + escape descending neurons -> takeoff and direction
    walking:   DNp09 (forward), MDN (backward), DNa02 (steering)

All tunable numbers are at the top of the file. Both the server (02_server.py)
and the calibration (04_calibrate.py) use it, so they stay aligned.
"""

import copy
import json
import math

import numpy as np

from flysim import PROC

# ---------------------------------------------------------------- neurons
LOOM_TYPES = ["LC4", "LPLC2"]                     # "something is coming" detectors
GIANT_FIBER = "DNp01"
ESCAPE_POOL = ["DNp01", "DNp02", "DNp04", "DNp11"]

WALK_FWD = ["DNp09"]                              # P9: forward walking
WALK_BACK = ["MDN"]                               # moonwalker: backward walking
WALK_TURN = "DNa02"                               # steering towards its own side

# Game odors -> glomeruli (each glomerulus = one ORN type in the antennae)
ODORS = {
    "fruity": ["ORN_DM1", "ORN_DM2", "ORN_DM4", "ORN_VM2"],   # fruit, vinegar: attractive
    "rotten": ["ORN_DA2", "ORN_V"],                           # geosmin, CO2: aversive
}

NOISE_SUPERCLASS = "sensory"                      # who receives the spontaneous activity

# Taste: sugar receptors and MN9 (proboscis), the same as in Shiu's paper
# (IDs from the example notebook at github.com/philshiu/Drosophila_brain_model)
SUGAR_ROOT_IDS = [
    720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345,
    720575940617000768, 720575940630797113, 720575940632889389, 720575940621754367,
    720575940621502051, 720575940640649691, 720575940639332736, 720575940616885538,
    720575940639198653, 720575940620900446, 720575940617937543, 720575940632425919,
    720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
    720575940611875570,
]
MN9_TYPE = "CB0701"        # this is how MN9 is annotated in FlyWire
TASTE_RATE = 100.0         # Hz on the sugar receptors (as in the paper)
FEED_HZ = 20.0             # above this mean MN9 rate the fly starts eating
FEED_STOP_HZ = 8.0         # ...and stops only when the mean drops below this
# Touch: a hit stimulates the head bristles and the antennal neurons that sense
# deflections (JO-E). The bristles always activate the same descending neurons (measured with
# 14_touch.py): we use them as the "grooming" output, the real fly's response to touch.
TOUCH_PREFIXES = ["BM_", "JO-E"]
TOUCH_RATE = 150.0         # Hz on the touch neurons for a full hit
GROOM_TYPES = ["DNg84", "DNg35", "DNg15", "DNge132", "DNg48"]
GROOM_HZ = 40.0            # above this mean rate of the descending neurons: it grooms
GROOM_STOP_HZ = 12.0
GROOM_SMOOTH = 0.15

FEED_SMOOTH = 0.1          # MN9 fires in bursts: the proboscis muscle integrates them
                           # (moving average of about half a second)

# Perception: for each odor we read the neurons most strongly connected downstream of its ORNs
# (antennal lobe projection neurons and interneurons). The connectome finds them, not a name.
PERCEIVE_TOP_K = 30
PERCEIVE_CALIB_MS = 300.0  # at startup: how long to simulate the "full" odor to calibrate the scale

# ---------------------------------------------------------------- senses
PLAYER_RADIUS = 0.3        # half the width of a player (blocks)
LOOM_RATE = 150.0          # maximum Hz on the looming detectors
LOOM_MIN = 0.25            # expansion rad/s below which the eye does not react (threshold)
LOOM_FULL = 2.0            # retinal expansion rad/s that gives intensity 1
EYE_OVERLAP = 20.0         # frontal degrees seen by both eyes
MAX_RANGE = 16.0           # beyond this distance the fly cannot see

# Escape habituation: in the real fly, repeated looming that does no harm triggers
# fewer and fewer escapes, because the looming neurons (LC4) respond less and less; a hit
# makes it sensitive again. It is a gain on the looming detectors, one per fly.
HAB_RATE = 2.0             # how fast it habituates (per unit of looming per second)
HAB_MIN = 0.05             # minimum gain: once habituated it ignores nearby steps, but someone running at it still makes it flee
HAB_RECOVER_S = 120.0      # seconds to become sensitive again without looming
HAB_PAIN = 0.3             # pain at least this strong makes it sensitive again immediately

ODOR_RATE = 80.0           # maximum Hz on the ORNs
ODOR_D0 = 3.0              # at this distance the concentration is half
ODOR_LAT = 0.4             # how much the side matters (0 = both antennae equal, 0.5 = all on one side)

# ---------------------------------------------------------------- motors
WALK_FULL_HZ = 40.0        # forward-backward difference (Hz) that counts as "full speed"
TURN_FULL_HZ = 40.0        # left-right DNa02 difference (Hz) that counts as "full steering"


# Player odors: each player has its own odor, chosen by 17_player_odors.py among those
# the brain tells apart best. The receptors saturate: at PLAYER_HALF concentration
# they already give half the response (real ORNs respond sublinearly).
PLAYER_HALF = 0.3


def player_rate(conc):
    """Hz on the ORNs of a player odor at concentration conc (0..1), intensity 1."""
    conc = max(0.0, min(1.0, float(conc)))
    return ODOR_RATE * conc / (conc + PLAYER_HALF) * (1 + PLAYER_HALF)


class PlayerOdors:
    """The players' odors. The odors are chosen by 17_player_odors.py (the ones the brain
    tells apart best); here each player gets its own the first time a fly smells it,
    and keeps it forever (data/processed/player_assign.json). It is the player's odor, not
    the fly's: all flies smell it the same way."""

    FILE = PROC / "player_odors.json"
    ASSIGN = PROC / "player_assign.json"

    def __init__(self, brain, sides):
        self.odors = []                         # per odor: {"L": (idx, intensity), "R": (...)}
        if self.FILE.exists():
            for o in json.loads(self.FILE.read_text(encoding="utf-8")):
                od = {}
                for s in "LR":
                    idx = [brain.find(g, sides[s]) for g in o["glomeruli"]]
                    w = [np.full(len(i), x, np.float32) for i, x in zip(idx, o["profile"])]
                    od[s] = (np.concatenate(idx), np.concatenate(w))
                self.odors.append(od)
        try:
            self.assign = json.loads(self.ASSIGN.read_text(encoding="utf-8")) if self.ASSIGN.exists() else {}
        except (OSError, ValueError):
            self.assign = {}

    def index(self, player):
        """Index of a player's odor (assigned if the player is new), or None if there are no odors."""
        if not self.odors:
            return None
        i = self.assign.get(player)
        if i is None or i >= len(self.odors):
            used = [0] * len(self.odors)
            for v in self.assign.values():
                if 0 <= v < len(used):
                    used[v] += 1
            i = used.index(min(used))                  # the least used one: the first players all get different odors
            self.assign[player] = i
            try:
                self.ASSIGN.write_text(json.dumps(self.assign, indent=1), encoding="utf-8")
            except OSError:
                pass
        return i

    def all_idx(self):
        if not self.odors:
            return np.array([], np.int64)
        return np.unique(np.concatenate([o[s][0] for o in self.odors for s in "LR"]))


def _rate(delta, idx, window_s):
    """Mean rate (Hz) of a group over the window."""
    if len(idx) == 0:
        return 0.0
    return float(delta[idx].sum()) / len(idx) / window_s


class FlyCircuits:
    def __init__(self, brain, swap_sides=False, protect_escape=True):
        self.brain = brain
        self.sides = {"L": "right", "R": "left"} if swap_sides else {"L": "left", "R": "right"}
        cat = lambda types, s: (np.concatenate([brain.find(t, self.sides[s]) for t in types])
                                if types else np.array([], np.int64))

        self.eye = {s: cat(LOOM_TYPES, s) for s in "LR"}
        self.gf = {s: brain.find(GIANT_FIBER, self.sides[s]) for s in "LR"}
        self.escape = {s: cat(ESCAPE_POOL, s) for s in "LR"}
        self.fwd = np.concatenate([cat(WALK_FWD, s) for s in "LR"])
        self.back = np.concatenate([cat(WALK_BACK, s) for s in "LR"])
        self.turn = {s: brain.find(WALK_TURN, self.sides[s]) for s in "LR"}
        self.odor = {name: {s: cat(types, s) for s in "LR"} for name, types in ODORS.items()}
        self.noise = brain.find_superclass(NOISE_SUPERCLASS)
        self.players = PlayerOdors(brain, self.sides)   # shared between the flies (sibling)

        roots = brain.neurons["root_id"] if "root_id" in brain.neurons else None
        self.sugar = (brain.neurons["idx"].to_numpy()[roots.isin(SUGAR_ROOT_IDS).to_numpy()]
                      if roots is not None else np.array([], np.int64))
        self.mn9 = brain.find(MN9_TYPE)
        self.mn9_avg = 0.0
        self.feeding = False

        self.touch = {s: np.concatenate([brain.find_prefix(p, self.sides[s]) for p in TOUCH_PREFIXES])
                      for s in "LR"}
        self.groom = np.concatenate([brain.find(t) for t in GROOM_TYPES])
        self.groom_avg = 0.0
        self.grooming = False
        self.loom_gain = 1.0                             # habituation (1 = sensitive)

        # the escape reflex is the part that works: no adaptation there
        if protect_escape:
            brain.exempt_adaptation(np.concatenate(
                [self.eye["L"], self.eye["R"], self.escape["L"], self.escape["R"]]))

        all_orn = brain.find_prefix("ORN")
        self.perceive_idx = {
            name: self._downstream(np.concatenate([d["L"], d["R"]]), exclude=all_orn)
            for name, d in self.odor.items()
        }
        # neurons downstream of several odors (shared interneurons) do not tell odors apart: drop them
        orig = dict(self.perceive_idx)
        for a in orig:
            others = [orig[b] for b in orig if b != a]
            if others:
                self.perceive_idx[a] = orig[a][~np.isin(orig[a], np.concatenate(others))]
        self.perceive_full = {name: 1.0 for name in self.odor}   # Hz of a "full odor"

    def _downstream(self, src, exclude, k=PERCEIVE_TOP_K):
        """The k neurons that receive the most excitatory synapses from the group src."""
        b = self.brain
        acc = np.zeros(b.N, np.float32)
        for j in src:
            s, e = b.indptr[j], b.indptr[j + 1]
            np.add.at(acc, b.indices[s:e], np.maximum(b.data[s:e], 0))
        acc[exclude] = 0
        top = np.argsort(acc)[::-1][:k]
        return top[acc[top] > 0]

    def calibrate_perception(self, ms=PERCEIVE_CALIB_MS):
        """Measures how strongly each odor activates at maximum, to get a 0..1 scale."""
        none = {"L": 0.0, "R": 0.0}
        for name in self.odor:
            smell = {n: ({"L": 1.0, "R": 1.0} if n == name else dict(none)) for n in self.odor}
            self.brain.reset()
            self.apply(dict(none), smell)
            self.brain.run(ms)
            rate = _rate(self.brain.counts, self.perceive_idx[name], ms / 1000)
            self.perceive_full[name] = max(rate, 1.0)
        self.brain.reset()
        return dict(self.perceive_full)

    # ------------------------------------------------------------ summary
    def sibling(self, brain):
        """The same circuits wired to another brain (see FlyBrain.sibling)."""
        c = copy.copy(self)
        c.brain = brain
        c.mn9_avg, c.feeding = 0.0, False
        c.groom_avg, c.grooming = 0.0, False
        c.loom_gain = 1.0
        return c

    def report(self):
        n = lambda d: f"{len(d['L'])} + {len(d['R'])}"
        lines = [
            f"Eyes (looming):         {n(self.eye)}",
            f"Giant Fiber:            {n(self.gf)}",
            f"Escape descending:      {n(self.escape)}",
            f"Walk forward:           {len(self.fwd)}",
            f"Walk backward:          {len(self.back)}",
            f"Steering (DNa02):       {n(self.turn)}",
        ]
        for name, d in self.odor.items():
            lines.append(f"Odor '{name}': {' ' * max(1, 15 - len(name))}{n(d)} ORN, "
                         f"perception read from {len(self.perceive_idx[name])} downstream neurons")
        lines.append(f"Background noise on:    {len(self.noise)} sensory neurons")
        lines.append(f"Player odors:           {len(self.players.odors)} distinct"
                     + (f", {len(self.players.assign)} players already known" if self.players.odors else
                        " (data/processed/player_odors.json missing: run 17_player_odors.py)"))
        lines.append(f"Touch:                  {len(self.touch['L'])} + {len(self.touch['R'])} neurons "
                     f"(bristles and antennae), grooming read from {len(self.groom)} descending neurons")
        lines.append(f"Taste:                  {len(self.sugar)} sugar receptors, "
                     f"{len(self.mn9)} MN9")
        warnings = []
        if len(self.sugar) == 0 or len(self.mn9) == 0:
            warnings.append("taste unavailable (sugar receptors or MN9 not found)")
        if len(self.fwd) == 0:
            warnings.append("no neurons for forward walking")
        if len(self.turn["L"]) == 0 or len(self.turn["R"]) == 0:
            warnings.append("DNa02 missing on one side: no steering")
        for name, d in self.odor.items():
            if len(d["L"]) + len(d["R"]) == 0:
                warnings.append(f"no ORN found for odor '{name}'")
        return lines, warnings

    def set_noise(self, rate_hz):
        self.brain.set_background(self.noise, rate_hz)

    # ---------------------------------------------------------------- senses
    @staticmethod
    def _norm_angle(a):
        return (float(a) + 180.0) % 360.0 - 180.0

    def looming(self, threats):
        """Looming intensity (0..1) per eye."""
        out = {"L": 0.0, "R": 0.0}
        for th in threats or []:
            d, v = float(th["dist"]), float(th["speed"])
            if v <= 0 or d > MAX_RANGE:
                continue
            d = max(d, 0.1)
            dtheta = 2 * PLAYER_RADIUS * v / (d * d + PLAYER_RADIUS ** 2)
            inten = min(1.0, max(0.0, dtheta - LOOM_MIN) / (LOOM_FULL - LOOM_MIN))
            if inten <= 0:
                continue
            a = self._norm_angle(th["angle"])
            if a > -EYE_OVERLAP:
                out["L"] = max(out["L"], inten)
            if a < EYE_OVERLAP:
                out["R"] = max(out["R"], inten)
        return out

    def smell(self, odors):
        """Concentration (0..1) of each odor per antenna. The player odors are in
        out["players"]: {odor index: {"L", "R"}}."""
        out = {name: {"L": 0.0, "R": 0.0} for name in ODORS}
        out["players"] = {}
        for src in odors or []:
            name = src.get("odor")
            if name == "player":
                i = self.players.index(str(src.get("player", "")))
                if i is None:
                    continue
                dst = out["players"].setdefault(i, {"L": 0.0, "R": 0.0})
            elif name in ODORS:
                dst = out[name]
            else:
                continue
            d = max(0.0, float(src["dist"]))
            if d > MAX_RANGE:
                continue
            c = 1.0 / (1.0 + (d / ODOR_D0) ** 2)
            s = math.sin(math.radians(self._norm_angle(src["angle"])))   # +1 = on the left
            dst["L"] = min(1.0, dst["L"] + c * (0.5 + ODOR_LAT * s) * 2)
            dst["R"] = min(1.0, dst["R"] + c * (0.5 - ODOR_LAT * s) * 2)
        return out

    def apply(self, loom, smell, taste=None, touch=None):
        """Turns the senses into Poisson stimuli on the right neurons."""
        idx, rates = [], []
        for s in "LR":
            hit = float((touch or {}).get(s, 0.0))
            if hit > 0 and len(self.touch[s]):
                idx.append(self.touch[s])
                rates.append(np.full(len(self.touch[s]), min(1.0, hit) * TOUCH_RATE, np.float32))
        sweet = float((taste or {}).get("sweet", 0.0))
        if sweet > 0 and len(self.sugar):
            idx.append(self.sugar)
            rates.append(np.full(len(self.sugar), min(1.0, sweet) * TASTE_RATE, np.float32))
        for s in "LR":
            idx.append(self.eye[s])
            rates.append(np.full(len(self.eye[s]), loom[s] * LOOM_RATE * self.loom_gain, np.float32))
            for name, d in self.odor.items():
                idx.append(d[s])
                rates.append(np.full(len(d[s]), smell[name][s] * ODOR_RATE, np.float32))
            for i, conc in smell.get("players", {}).items():
                if conc[s] > 0:
                    pidx, w = self.players.odors[i][s]
                    idx.append(pidx)
                    rates.append((player_rate(conc[s]) * w).astype(np.float32))
        self.brain.set_stimulus(np.concatenate(idx), np.concatenate(rates))

    def habituate(self, loom, pain, window_ms):
        """After each window: seen looming habituates the fly, calm and pain make it sensitive again."""
        dt = window_ms / 1000.0
        if pain >= HAB_PAIN:
            self.loom_gain = 1.0
            return
        seen = max(loom.get("L", 0.0), loom.get("R", 0.0))
        self.loom_gain *= math.exp(-HAB_RATE * seen * dt)
        self.loom_gain += (1.0 - self.loom_gain) * dt / HAB_RECOVER_S
        self.loom_gain = max(HAB_MIN, min(1.0, self.loom_gain))

    # --------------------------------------------------------------- motors
    def decode(self, delta, window_ms):
        w = window_ms / 1000.0
        gf = [int(delta[self.gf["L"]].sum()), int(delta[self.gf["R"]].sum())]

        # escape: the more active side of the descending neurons is the side of the threat
        pl, pr = float(delta[self.escape["L"]].sum()), float(delta[self.escape["R"]].sum())
        lat = (pl - pr) / (pl + pr) if pl + pr > 0 else 0.0
        efwd, eleft = -(1 - abs(lat)), -lat
        en = math.hypot(efwd, eleft) or 1.0

        # walking
        f_hz = _rate(delta, self.fwd, w)
        b_hz = _rate(delta, self.back, w)
        tl_hz = _rate(delta, self.turn["L"], w)
        tr_hz = _rate(delta, self.turn["R"], w)
        walk_fwd = max(-0.5, min(1.0, (f_hz - b_hz) / WALK_FULL_HZ))
        walk_turn = max(-1.0, min(1.0, (tl_hz - tr_hz) / TURN_FULL_HZ))   # + = left

        # perception: how strongly the brain "smells" each odor (0..1)
        perceive = {name: round(min(1.0, _rate(delta, idx, w) / self.perceive_full[name]), 3)
                    for name, idx in self.perceive_idx.items()}

        mn9_hz = _rate(delta, self.mn9, w)
        self.mn9_avg += (mn9_hz - self.mn9_avg) * FEED_SMOOTH
        self.feeding = self.mn9_avg >= (FEED_STOP_HZ if self.feeding else FEED_HZ)
        groom_hz = _rate(delta, self.groom, w)
        self.groom_avg += (groom_hz - self.groom_avg) * GROOM_SMOOTH
        self.grooming = self.groom_avg >= (GROOM_STOP_HZ if self.grooming else GROOM_HZ)

        return {
            "spikes": int(delta.sum()),                  # spikes in the whole brain (debug)
            "mn9": round(mn9_hz, 1),
            "mn9_avg": round(self.mn9_avg, 1),
            "feeding": self.feeding,
            "groom": round(self.groom_avg, 1),
            "grooming": self.grooming,
            "perceive": perceive,
            "gf": gf,
            "takeoff": sum(gf) > 0,
            "escape": {"fwd": round(efwd / en, 3), "left": round(eleft / en, 3)},
            "walk": {"fwd": round(walk_fwd, 3), "turn": round(walk_turn, 3)},
            "dn": {"fwd": round(f_hz, 1), "back": round(b_hz, 1),
                   "turnL": round(tl_hz, 1), "turnR": round(tr_hz, 1)},
        }
