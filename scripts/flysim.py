r"""
flysim - motore di simulazione del cervello della mosca
--------------------------------------------------------
Rete leaky integrate-and-fire (LIF) sul connettoma FlyWire v783,
con i parametri del modello di Shiu et al. (Nature, 2024).

Simulazione event-driven: a ogni passo propagano solo i neuroni che sparano.
Due motori CPU con lo stesso risultato:
  - Numba (se installato: pip install numba): compilato, molto più veloce
    quando tanti neuroni sono attivi
  - NumPy: di riserva

Due tipi di input:
  - stimolo:  Poisson su gruppi scelti (occhi, antenne...), cambia ogni finestra
  - rumore:   attività spontanea di fondo, fissa (il mondo non è mai silenzioso)
"""

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:                                    # motore NumPy di riserva
    HAVE_NUMBA = False

# Motore Numba: sotto queste soglie (mV) un neurone torna "a riposo" ed esce dai calcoli.
# Sono minuscole rispetto ai 7 mV che servono per sparare.
X_REST = 0.01
V_REST = 0.05

if HAVE_NUMBA:
    @njit(cache=True)
    def _seed(s):
        np.random.seed(s)

    @njit(cache=True, nogil=True)
    def _wake(i, act, alist, nact):
        if act[i] == 0:
            act[i] = 1
            alist[nact] = i
            nact += 1
        return nact

    @njit(cache=True, nogil=True)
    def _run_numba(nsteps, t, v, x, a, ref_until, ring, mark, tlist, tcount,
                   act, alist, nact_box, D, ref_steps, decay_s, decay_a, adapt_b, k_m,
                   v0, vr, vth, poi_w, indptr, indices, data, counts,
                   stim_idx, stim_p, bg_idx, bg_p):
        """Come step() ma lavora solo sui neuroni attivi: a riposo costa quasi zero.
        ring/mark/tlist: per ogni passo di ritardo, chi riceverà input e quanto.
        act/alist: i neuroni non a riposo (corrente o potenziale diversi da zero)."""
        nact = nact_box[0]
        nb = bg_idx.shape[0]
        for _ in range(nsteps):
            slot = t % D
            r = ring[slot]
            mk = mark[slot]
            tl = tlist[slot]
            # input sinaptico in arrivo (spike di D passi fa)
            for k in range(tcount[slot]):
                i = tl[k]
                x[i] += r[i]
                r[i] = 0.0
                mk[i] = 0
                nact = _wake(i, act, alist, nact)
            tcount[slot] = 0
            # stimoli esterni
            for k in range(stim_idx.shape[0]):
                if np.random.random() < stim_p[k]:
                    i = stim_idx[k]
                    x[i] += poi_w
                    nact = _wake(i, act, alist, nact)
            # rumore di fondo
            if nb > 0 and bg_p > 0.0:
                n = np.random.binomial(nb, bg_p)
                for _k in range(n):
                    i = bg_idx[np.random.randint(0, nb)]
                    x[i] += poi_w
                    nact = _wake(i, act, alist, nact)
            # membrana e spike, solo per i neuroni attivi (la lista si compatta)
            w = 0
            for k in range(nact):
                i = alist[k]
                xi = x[i]
                vi = v[i]
                ai = a[i]
                if ref_until[i] <= t:
                    vi += k_m * (xi - (vi - v0) - ai)
                xi *= decay_s
                ai *= decay_a
                # sotto queste soglie il neurone è "a riposo" (la soglia di spike è 7 mV sopra)
                if -X_REST < xi < X_REST:
                    xi = 0.0
                if ai < X_REST:
                    ai = 0.0
                if -V_REST < vi - v0 < V_REST:
                    vi = v0
                if vi >= vth:
                    vi = vr
                    ai += adapt_b[i]               # adattamento: ogni spike "stanca" il neurone
                    ref_until[i] = t + ref_steps
                    counts[i] += 1
                    for p in range(indptr[i], indptr[i + 1]):  # arriverà tra D passi
                        j = indices[p]
                        if mk[j] == 0:
                            mk[j] = 1
                            tl[tcount[slot]] = j
                            tcount[slot] += 1
                        r[j] += data[p]
                x[i] = xi
                v[i] = vi
                a[i] = ai
                if xi == 0.0 and vi == v0 and ai == 0.0:
                    act[i] = 0                     # tornato a riposo: esce dalla lista
                else:
                    alist[w] = i
                    w += 1
            nact = w
            t += 1
        nact_box[0] = nact
        return t

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"

# Parametri Shiu et al. 2024 (tempi in ms, potenziali in mV)
DEFAULT_PARAMS = dict(
    v0=-52.0,      # potenziale di riposo
    vr=-52.0,      # potenziale di reset dopo lo spike
    vth=-45.0,     # soglia di spike
    tau_m=20.0,    # costante di tempo di membrana
    tau_s=5.0,     # costante di tempo sinaptica
    t_ref=2.2,     # periodo refrattario
    delay=1.8,     # ritardo sinaptico
    w_syn=0.275,   # peso di una singola sinapsi
    f_poi=250.0,   # moltiplicatore del peso per gli stimoli di Poisson
    # Adattamento (non nel modello di Shiu, spento di default): ogni spike aggiunge
    # adapt_b mV di "stanchezza" che si smaltisce con costante tau_a. Impedisce che
    # l'attività si autoalimenti all'infinito.
    adapt_b=0.0,
    tau_a=200.0,
    # Sinapsi tra cellule di Kenyon (corpo a fungo). Nel connettoma sono tante, ma nella
    # mosca vera le KC restano silenziose; nel modello si accendono a vicenda all'infinito.
    # False = le tagliamo (solo KC -> KC, il resto del corpo a fungo resta).
    kc_recurrent=True,
    # Interneuroni locali del lobo antennale (lLN1_bc, il3LN6, ...) a cui il connettoma
    # assegna un modulatore (dopamina, serotonina, octopamina): la stima è poco affidabile
    # e nella mosca vera sono in gran parte GABAergici. True = li rendiamo inibitori.
    fix_ln_modulatory=False,
    # Neuromodulatori (octopamina, dopamina, serotonina) agiscono lentamente tramite
    # recettori metabotropici, non come eccitazione rapida. Il modello di Shiu li tratta
    # come eccitatori veloci. "keep" = come Shiu; "oa" = solo i neuroni octopaminergici
    # per nome (OA-...) non danno eccitazione rapida; "all" = nessun neuromodulatore
    # (per nome o per stima) dà eccitazione rapida, tranne le cellule di Kenyon
    # (stimate dopamina ma colinergiche) e gli interneuroni già corretti.
    modulators="keep",
)

# Configurazione consigliata, scelta con gli script di diagnosi (07-13):
# il modello di Shiu con tre errori del connettoma corretti (interneuroni del lobo
# antennale, sinapsi KC->KC, neuromodulatori trattati come eccitazione rapida) più un
# adattamento leggero. Fuga e gusto (zucchero -> MN9) come nell'originale, olfatto
# stabile: il cervello si calma quando l'odore se ne va.
RECOMMENDED = dict(fix_ln_modulatory=True, kc_recurrent=False, modulators="all",
                   adapt_b=0.5, tau_a=300.0)


class FlyBrain:
    def __init__(self, dt=0.1, params=None, seed=None, neurons=None, weights=None):
        self.p = {**DEFAULT_PARAMS, **(params or {})}
        self.dt = dt
        self.rng = np.random.default_rng(seed)

        self.neurons = neurons if neurons is not None else pd.read_parquet(PROC / "neurons.parquet")
        W = (weights if weights is not None else sp.load_npz(PROC / "weights.npz")).tocsc()
        self.N = W.shape[0]
        self.indptr = W.indptr.astype(np.int64)
        self.indices = W.indices
        self.data = (W.data * self.p["w_syn"]).astype(np.float32)

        self.D = max(1, round(self.p["delay"] / dt))       # passi di ritardo
        self.ref_steps = round(self.p["t_ref"] / dt)
        self.decay_s = np.float32(np.exp(-dt / self.p["tau_s"]))
        self.decay_a = np.float32(np.exp(-dt / self.p["tau_a"]))
        # adattamento per neurone (0 = esente, vedi exempt_adaptation)
        self.adapt_b = np.full(self.N, self.p["adapt_b"], np.float32)
        self.k_m = np.float32(dt / self.p["tau_m"])
        self.poi_w = np.float32(self.p["w_syn"] * self.p["f_poi"])

        self._ct = self.neurons["cell_type"].fillna("").astype(str).str.upper()

        self.kc_cut = 0
        if not self.p["kc_recurrent"]:
            kc = self._ct.str.startswith("KC").to_numpy()
            col = np.repeat(np.arange(self.N), np.diff(self.indptr))   # presinaptico di ogni sinapsi
            cut = kc[col] & kc[self.indices]
            self.data[cut] = 0.0
            self.kc_cut = int(cut.sum())

        self.ln_fixed = 0
        if self.p["fix_ln_modulatory"]:
            nt = self.neurons["top_nt"].fillna("").astype(str).str.lower()
            names = self.neurons["cell_type"].fillna("").astype(str)
            is_ln = names.str.match(r"^[a-z][a-z0-9]*LN").to_numpy()            # lLN1_bc, il3LN6, v2LN...
            modulatory = nt.isin(["dopamine", "serotonin", "octopamine"]).to_numpy()
            fix = is_ln & modulatory
            col = np.repeat(np.arange(self.N), np.diff(self.indptr))
            sel = fix[col]
            self.data[sel] = -np.abs(self.data[sel])                       # da eccitatori a inibitori
            self.ln_fixed = int(fix.sum())

        self.mod_silenced = 0
        mode = self.p["modulators"]
        if mode in ("oa", "all"):
            names = self.neurons["cell_type"].fillna("").astype(str)
            nt = self.neurons["top_nt"].fillna("").astype(str).str.lower()
            mod = names.str.startswith("OA-").to_numpy()
            if mode == "all":
                by_name = names.str.match(r"^(OA-|5-HT|PAM\d|PPL\d|PPM\d|CSD)").to_numpy()
                by_nt = nt.isin(["dopamine", "serotonin", "octopamine"]).to_numpy()
                is_kc = names.str.startswith("KC").to_numpy()
                is_ln = names.str.match(r"^[a-z][a-z0-9]*LN").to_numpy()
                mod = (by_name | by_nt) & ~is_kc & ~(is_ln & self.p["fix_ln_modulatory"])
            col = np.repeat(np.arange(self.N), np.diff(self.indptr))
            sel = mod[col] & (self.data > 0)                  # tolgo solo l'eccitazione rapida
            self.data[sel] = 0.0
            self.mod_silenced = int(mod.sum())
        self._hb = self.neurons["hemibrain_type"].fillna("").astype(str).str.upper()
        self._side = self.neurons["side"].fillna("").astype(str)
        self._idx = self.neurons["idx"].to_numpy()

        self.bg_idx = np.array([], np.int64)
        self.bg_p = 0.0
        self.engine = "numba" if HAVE_NUMBA else "numpy"
        if HAVE_NUMBA:
            _seed(int(self.rng.integers(0, 2**31 - 1)))
        self.reset()

    # ------------------------------------------------------------------ stato
    def sibling(self, seed=None):
        """Un altro cervello con lo stesso connettoma (pesi e anagrafica condivisi,
        nessuna copia in memoria) ma con il suo stato: serve per avere più mosche."""
        b = copy.copy(self)
        b.rng = np.random.default_rng(seed)
        b.reset()                                    # potenziali, correnti, ritardi: tutti nuovi
        return b

    def reset(self):
        """Riporta il cervello a riposo. Il rumore di fondo resta impostato."""
        N = self.N
        self.v = np.full(N, self.p["v0"], np.float32)
        self.x = np.zeros(N, np.float32)                   # corrente sinaptica
        self.a = np.zeros(N, np.float32)                   # adattamento
        self.ref_until = np.zeros(N, np.int64)
        self.ring = np.zeros((self.D, N), np.float32)      # buffer del ritardo
        # strutture del motore Numba (vedi _run_numba)
        self.mark = np.zeros((self.D, N), np.uint8)
        self.tlist = np.zeros((self.D, N), np.int32)
        self.tcount = np.zeros(self.D, np.int64)
        self.act = np.zeros(N, np.uint8)
        self.alist = np.zeros(N, np.int32)
        self.nact = np.zeros(1, np.int64)
        self.t = 0
        self.counts = np.zeros(N, np.int32)
        self.clear_stimulus()

    # -------------------------------------------------------------- ricerca
    def _pick(self, mask, side=None):
        if side:
            mask = mask & (self._side == side)
        return self._idx[mask.to_numpy()]

    def find(self, cell_type, side=None):
        """Neuroni di un tipo cellulare esatto (opzionale: 'left'/'right')."""
        t = cell_type.upper()
        return self._pick((self._ct == t) | (self._hb == t), side)

    def find_prefix(self, prefix, side=None):
        """Neuroni il cui tipo inizia con un prefisso (es. 'ORN_')."""
        return self._pick(self._ct.str.startswith(prefix.upper()), side)

    def find_superclass(self, super_class, side=None):
        sc = self.neurons["super_class"].fillna("").astype(str).str.lower()
        return self._pick(sc == super_class.lower(), side)

    def types_with_prefix(self, prefix):
        """Tipi cellulari distinti che iniziano con un prefisso, con quanti neuroni."""
        names = self.neurons["cell_type"].fillna("").astype(str)
        hit = names[names.str.upper().str.startswith(prefix.upper())]
        return hit.value_counts().sort_index()

    def describe(self, i):
        r = self.neurons.iloc[i]
        ct = r["cell_type"] if isinstance(r["cell_type"], str) else "?"
        return f"{ct} ({r['side']}, {r['super_class']})"

    def exempt_adaptation(self, idx):
        """Questi neuroni non si stancano (es. il circuito della fuga)."""
        self.adapt_b[np.asarray(idx, np.int64)] = 0.0

    # ------------------------------------------------------------ input
    def clear_stimulus(self):
        self.stim_idx = np.array([], np.int64)
        self.stim_p = np.array([], np.float32)

    def set_stimulus(self, idx, rate_hz):
        """Stimolo di Poisson su un gruppo di neuroni (sostituisce quello attuale).
        rate_hz può essere un numero unico o un array, uno per neurone."""
        idx = np.asarray(idx, np.int64)
        rates = np.broadcast_to(np.asarray(rate_hz, np.float32), idx.shape)
        keep = rates > 0
        self.stim_idx = idx[keep]
        self.stim_p = (rates[keep] * self.dt * 1e-3).astype(np.float32)

    def add_stimulus(self, idx, rate_hz):
        """Aggiunge uno stimolo a quello attuale (es. un neurone cliccato nell'esploratore)."""
        idx = np.asarray(idx, np.int64)
        if len(idx) == 0 or rate_hz <= 0:
            return
        p = np.full(len(idx), rate_hz * self.dt * 1e-3, np.float32)
        self.stim_idx = np.concatenate([self.stim_idx, idx])
        self.stim_p = np.concatenate([self.stim_p, p])

    def set_background(self, idx, rate_hz):
        """Attività spontanea: ogni neurone del gruppo spara in media rate_hz volte al secondo."""
        self.bg_idx = np.asarray(idx, np.int64)
        self.bg_p = float(rate_hz) * self.dt * 1e-3

    # ---------------------------------------------------------- simulazione
    def step(self):
        p = self.p
        slot = self.t % self.D

        # input sinaptico in arrivo (spike di D passi fa)
        self.x += self.ring[slot]
        self.ring[slot] = 0.0

        # stimoli esterni
        if len(self.stim_idx):
            hit = self.rng.random(len(self.stim_idx)) < self.stim_p
            if hit.any():
                self.x[self.stim_idx[hit]] += self.poi_w

        # rumore di fondo: quanti neuroni "scattano" in questo passo, poi quali
        nb = len(self.bg_idx)
        if nb and self.bg_p > 0:
            n = self.rng.binomial(nb, self.bg_p)
            if n:
                self.x[self.bg_idx[self.rng.integers(0, nb, n)]] += self.poi_w

        # membrana (ferma durante il periodo refrattario)
        dv = self.k_m * (self.x - (self.v - p["v0"]) - self.a)
        dv[self.ref_until > self.t] = 0.0
        self.v += dv
        self.x *= self.decay_s
        self.a *= self.decay_a

        # spike
        spk = np.flatnonzero(self.v >= p["vth"])
        if len(spk):
            self.v[spk] = p["vr"]
            self.a[spk] += self.adapt_b[spk]
            self.ref_until[spk] = self.t + self.ref_steps
            self.counts[spk] += 1
            self._deliver(spk, self.ring[slot])         # arriverà tra D passi

        self.t += 1
        return spk

    def _deliver(self, spk, ring):
        if len(spk) <= 4:                               # pochi spike: semplice
            for j in spk:
                s, e = self.indptr[j], self.indptr[j + 1]
                ring[self.indices[s:e]] += self.data[s:e]
            return
        # tanti spike: raccogliamo tutte le colonne in un colpo solo
        starts = self.indptr[spk]
        lens = self.indptr[spk + 1] - starts
        tot = int(lens.sum())
        if tot == 0:
            return
        offs = np.repeat(starts - (np.cumsum(lens) - lens), lens) + np.arange(tot)
        np.add.at(ring, self.indices[offs], self.data[offs])

    def run(self, ms):
        n = int(round(ms / self.dt))
        if HAVE_NUMBA:
            p = self.p
            self.t = _run_numba(
                n, self.t, self.v, self.x, self.a, self.ref_until, self.ring,
                self.mark, self.tlist, self.tcount, self.act, self.alist, self.nact,
                self.D, self.ref_steps, self.decay_s, self.decay_a, self.adapt_b, self.k_m,
                np.float32(p["v0"]), np.float32(p["vr"]), np.float32(p["vth"]), self.poi_w,
                self.indptr, self.indices, self.data, self.counts,
                self.stim_idx, self.stim_p, self.bg_idx, float(self.bg_p))
        else:
            for _ in range(n):
                self.step()
