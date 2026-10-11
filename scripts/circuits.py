r"""
circuits - come la mosca sente e come si muove
-----------------------------------------------
Qui sta tutta la "traduzione" tra il mondo di Minecraft e i neuroni:

  SENSI   (mondo -> neuroni)
    occhi:     player che si avvicinano  -> LC4, LPLC2      (looming)
    antenne:   odori degli item          -> ORN dei glomeruli olfattivi
    rumore:    attività spontanea         -> neuroni sensoriali

  MOTORI  (neuroni -> mondo)
    fuga:      Giant Fiber (DNp01) + discendenti di fuga -> decollo e direzione
    camminata: DNp09 (avanti), MDN (indietro), DNa02 (sterzata)

Tutti i numeri regolabili sono in cima al file. Lo usano sia il server
(02_server.py) sia la calibrazione (04_calibra.py), così restano allineati.
"""

import copy
import math

import numpy as np

# ---------------------------------------------------------------- neuroni
LOOM_TYPES = ["LC4", "LPLC2"]                     # rilevatori di "qualcosa che arriva"
GIANT_FIBER = "DNp01"
ESCAPE_POOL = ["DNp01", "DNp02", "DNp04", "DNp11"]

WALK_FWD = ["DNp09"]                              # P9: camminata in avanti
WALK_BACK = ["MDN"]                               # moonwalker: retromarcia
WALK_TURN = "DNa02"                               # sterzata verso il proprio lato

# Odori del gioco -> glomeruli (ogni glomerulo = un tipo di ORN nelle antenne)
ODORS = {
    "fruity": ["ORN_DM1", "ORN_DM2", "ORN_DM4", "ORN_VM2"],   # frutta, aceto: attraenti
    "rotten": ["ORN_DA2", "ORN_V"],                           # geosmina, CO2: avversivi
}

NOISE_SUPERCLASS = "sensory"                      # chi riceve l'attività spontanea

# Gusto: recettori dello zucchero e MN9 (proboscide), gli stessi del paper di Shiu
# (ID dal notebook di esempio di github.com/philshiu/Drosophila_brain_model)
SUGAR_ROOT_IDS = [
    720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345,
    720575940617000768, 720575940630797113, 720575940632889389, 720575940621754367,
    720575940621502051, 720575940640649691, 720575940639332736, 720575940616885538,
    720575940639198653, 720575940620900446, 720575940617937543, 720575940632425919,
    720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
    720575940611875570,
]
MN9_TYPE = "CB0701"        # così è annotato MN9 in FlyWire
TASTE_RATE = 100.0         # Hz sui recettori dello zucchero (come nel paper)
FEED_HZ = 20.0             # sopra questa frequenza media di MN9 la mosca inizia a mangiare
FEED_STOP_HZ = 8.0         # ...e smette solo quando la media scende sotto questa
# Tatto: un colpo stimola le setole della testa e i neuroni delle antenne che sentono le
# deflessioni (JO-E). Le setole accendono sempre gli stessi discendenti (misurati con
# 14_tatto.py): li usiamo come uscita "si pulisce", la risposta della mosca vera al tocco.
TOUCH_PREFIXES = ["BM_", "JO-E"]
TOUCH_RATE = 150.0         # Hz sui neuroni del tatto con un colpo pieno
GROOM_TYPES = ["DNg84", "DNg35", "DNg15", "DNge132", "DNg48"]
GROOM_HZ = 40.0            # sopra questa frequenza media dei discendenti: si pulisce
GROOM_STOP_HZ = 12.0
GROOM_SMOOTH = 0.15

FEED_SMOOTH = 0.1          # MN9 spara a raffiche: il muscolo della proboscide le integra
                           # (media mobile di circa mezzo secondo)

# Percezione: per ogni odore leggiamo i neuroni più collegati a valle dei suoi ORN
# (proiezione e interneuroni del lobo antennale). Li trova il connettoma, non un nome.
PERCEIVE_TOP_K = 30
PERCEIVE_CALIB_MS = 300.0  # all'avvio: quanto simulare l'odore "pieno" per tarare la scala

# ---------------------------------------------------------------- sensi
PLAYER_RADIUS = 0.3        # mezza larghezza di un player (blocchi)
LOOM_RATE = 150.0          # Hz massimi sui rilevatori di looming
LOOM_MIN = 0.25            # rad/s di espansione sotto cui l'occhio non reagisce (soglia)
LOOM_FULL = 2.0            # rad/s di espansione sulla retina che danno intensità 1
EYE_OVERLAP = 20.0         # gradi frontali visti da entrambi gli occhi
MAX_RANGE = 16.0           # oltre questa distanza la mosca non vede

ODOR_RATE = 80.0           # Hz massimi sugli ORN
ODOR_D0 = 3.0              # a questa distanza la concentrazione è metà
ODOR_LAT = 0.4             # quanto conta il lato (0 = antenne uguali, 0.5 = tutto da un lato)

# ---------------------------------------------------------------- motori
WALK_FULL_HZ = 40.0        # differenza avanti-indietro (Hz) che vale "velocità piena"
TURN_FULL_HZ = 40.0        # differenza DNa02 sinistra-destra (Hz) che vale "sterzata piena"


def _rate(delta, idx, window_s):
    """Frequenza media (Hz) di un gruppo nella finestra."""
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

        # il riflesso di fuga è la parte che funziona: niente adattamento lì
        if protect_escape:
            brain.exempt_adaptation(np.concatenate(
                [self.eye["L"], self.eye["R"], self.escape["L"], self.escape["R"]]))

        all_orn = brain.find_prefix("ORN")
        self.perceive_idx = {
            name: self._downstream(np.concatenate([d["L"], d["R"]]), exclude=all_orn)
            for name, d in self.odor.items()
        }
        # neuroni a valle di più odori (interneuroni comuni) non distinguono gli odori: via
        orig = dict(self.perceive_idx)
        for a in orig:
            others = [orig[b] for b in orig if b != a]
            if others:
                self.perceive_idx[a] = orig[a][~np.isin(orig[a], np.concatenate(others))]
        self.perceive_full = {name: 1.0 for name in self.odor}   # Hz di "odore pieno"

    def _downstream(self, src, exclude, k=PERCEIVE_TOP_K):
        """I k neuroni che ricevono più sinapsi eccitatorie dal gruppo src."""
        b = self.brain
        acc = np.zeros(b.N, np.float32)
        for j in src:
            s, e = b.indptr[j], b.indptr[j + 1]
            np.add.at(acc, b.indices[s:e], np.maximum(b.data[s:e], 0))
        acc[exclude] = 0
        top = np.argsort(acc)[::-1][:k]
        return top[acc[top] > 0]

    def calibrate_perception(self, ms=PERCEIVE_CALIB_MS):
        """Misura quanto si accende ogni odore al massimo, per avere una scala 0..1."""
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

    # ------------------------------------------------------------ resoconto
    def sibling(self, brain):
        """Gli stessi circuiti collegati a un altro cervello (vedi FlyBrain.sibling)."""
        c = copy.copy(self)
        c.brain = brain
        c.mn9_avg, c.feeding = 0.0, False
        c.groom_avg, c.grooming = 0.0, False
        return c

    def report(self):
        n = lambda d: f"{len(d['L'])} + {len(d['R'])}"
        lines = [
            f"Occhi (looming):        {n(self.eye)}",
            f"Giant Fiber:            {n(self.gf)}",
            f"Discendenti di fuga:    {n(self.escape)}",
            f"Camminata avanti:       {len(self.fwd)}",
            f"Camminata indietro:     {len(self.back)}",
            f"Sterzata (DNa02):       {n(self.turn)}",
        ]
        for name, d in self.odor.items():
            lines.append(f"Odore '{name}':{' ' * max(1, 15 - len(name))}{n(d)} ORN, "
                         f"percezione letta da {len(self.perceive_idx[name])} neuroni a valle")
        lines.append(f"Rumore di fondo su:     {len(self.noise)} neuroni sensoriali")
        lines.append(f"Tatto:                  {len(self.touch['L'])} + {len(self.touch['R'])} neuroni "
                     f"(setole e antenne), pulizia letta da {len(self.groom)} discendenti")
        lines.append(f"Gusto:                  {len(self.sugar)} recettori dello zucchero, "
                     f"{len(self.mn9)} MN9")
        warnings = []
        if len(self.sugar) == 0 or len(self.mn9) == 0:
            warnings.append("gusto non disponibile (zucchero o MN9 non trovati)")
        if len(self.fwd) == 0:
            warnings.append("nessun neurone per la camminata in avanti")
        if len(self.turn["L"]) == 0 or len(self.turn["R"]) == 0:
            warnings.append("manca DNa02 su un lato: niente sterzata")
        for name, d in self.odor.items():
            if len(d["L"]) + len(d["R"]) == 0:
                warnings.append(f"nessun ORN trovato per l'odore '{name}'")
        return lines, warnings

    def set_noise(self, rate_hz):
        self.brain.set_background(self.noise, rate_hz)

    # ---------------------------------------------------------------- sensi
    @staticmethod
    def _norm_angle(a):
        return (float(a) + 180.0) % 360.0 - 180.0

    def looming(self, threats):
        """Intensità di looming (0..1) per occhio."""
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
        """Concentrazione (0..1) di ogni odore per antenna."""
        out = {name: {"L": 0.0, "R": 0.0} for name in ODORS}
        for src in odors or []:
            name = src.get("odor")
            if name not in out:
                continue
            d = max(0.0, float(src["dist"]))
            if d > MAX_RANGE:
                continue
            c = 1.0 / (1.0 + (d / ODOR_D0) ** 2)
            s = math.sin(math.radians(self._norm_angle(src["angle"])))   # +1 = a sinistra
            out[name]["L"] = min(1.0, out[name]["L"] + c * (0.5 + ODOR_LAT * s) * 2)
            out[name]["R"] = min(1.0, out[name]["R"] + c * (0.5 - ODOR_LAT * s) * 2)
        return out

    def apply(self, loom, smell, taste=None, touch=None):
        """Trasforma i sensi in stimoli di Poisson sui neuroni giusti."""
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
            rates.append(np.full(len(self.eye[s]), loom[s] * LOOM_RATE, np.float32))
            for name, d in self.odor.items():
                idx.append(d[s])
                rates.append(np.full(len(d[s]), smell[name][s] * ODOR_RATE, np.float32))
        self.brain.set_stimulus(np.concatenate(idx), np.concatenate(rates))

    # --------------------------------------------------------------- motori
    def decode(self, delta, window_ms):
        w = window_ms / 1000.0
        gf = [int(delta[self.gf["L"]].sum()), int(delta[self.gf["R"]].sum())]

        # fuga: il lato più attivo dei discendenti è il lato della minaccia
        pl, pr = float(delta[self.escape["L"]].sum()), float(delta[self.escape["R"]].sum())
        lat = (pl - pr) / (pl + pr) if pl + pr > 0 else 0.0
        efwd, eleft = -(1 - abs(lat)), -lat
        en = math.hypot(efwd, eleft) or 1.0

        # camminata
        f_hz = _rate(delta, self.fwd, w)
        b_hz = _rate(delta, self.back, w)
        tl_hz = _rate(delta, self.turn["L"], w)
        tr_hz = _rate(delta, self.turn["R"], w)
        walk_fwd = max(-0.5, min(1.0, (f_hz - b_hz) / WALK_FULL_HZ))
        walk_turn = max(-1.0, min(1.0, (tl_hz - tr_hz) / TURN_FULL_HZ))   # + = sinistra

        # percezione: quanto il cervello "sente" ogni odore (0..1)
        perceive = {name: round(min(1.0, _rate(delta, idx, w) / self.perceive_full[name]), 3)
                    for name, idx in self.perceive_idx.items()}

        mn9_hz = _rate(delta, self.mn9, w)
        self.mn9_avg += (mn9_hz - self.mn9_avg) * FEED_SMOOTH
        self.feeding = self.mn9_avg >= (FEED_STOP_HZ if self.feeding else FEED_HZ)
        groom_hz = _rate(delta, self.groom, w)
        self.groom_avg += (groom_hz - self.groom_avg) * GROOM_SMOOTH
        self.grooming = self.groom_avg >= (GROOM_STOP_HZ if self.grooming else GROOM_HZ)

        return {
            "spikes": int(delta.sum()),                  # spike in tutto il cervello (debug)
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
