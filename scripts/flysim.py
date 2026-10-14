r"""
flysim - the fly brain simulation engine
----------------------------------------
Leaky integrate-and-fire (LIF) network on the FlyWire v783 connectome,
with the model parameters of Shiu et al. (Nature, 2024).

Event-driven simulation: at each step only the neurons that spike propagate.
Two CPU engines with the same result:
  - Numba (if installed: pip install numba): compiled, much faster
    when many neurons are active
  - NumPy: fallback

Two kinds of input:
  - stimulus:   Poisson on chosen groups (eyes, antennae...), changes every window
  - noise:      spontaneous background activity, fixed (the world is never silent)
"""

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:                                    # NumPy fallback engine
    HAVE_NUMBA = False

# Numba engine: below these thresholds (mV) a neuron goes back "to rest" and leaves the computation.
# They are tiny compared with the 7 mV needed to spike.
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
        """Like step() but works only on the active neurons: at rest it costs almost nothing.
        ring/mark/tlist: for each delay step, who will receive input and how much.
        act/alist: the neurons not at rest (non-zero current or potential)."""
        nact = nact_box[0]
        nb = bg_idx.shape[0]
        for _ in range(nsteps):
            slot = t % D
            r = ring[slot]
            mk = mark[slot]
            tl = tlist[slot]
            # incoming synaptic input (spikes from D steps ago)
            for k in range(tcount[slot]):
                i = tl[k]
                x[i] += r[i]
                r[i] = 0.0
                mk[i] = 0
                nact = _wake(i, act, alist, nact)
            tcount[slot] = 0
            # external stimuli
            for k in range(stim_idx.shape[0]):
                if np.random.random() < stim_p[k]:
                    i = stim_idx[k]
                    x[i] += poi_w
                    nact = _wake(i, act, alist, nact)
            # background noise
            if nb > 0 and bg_p > 0.0:
                n = np.random.binomial(nb, bg_p)
                for _k in range(n):
                    i = bg_idx[np.random.randint(0, nb)]
                    x[i] += poi_w
                    nact = _wake(i, act, alist, nact)
            # membrane and spikes, only for the active neurons (the list is compacted)
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
                # below these thresholds the neuron is "at rest" (the spike threshold is 7 mV above)
                if -X_REST < xi < X_REST:
                    xi = 0.0
                if ai < X_REST:
                    ai = 0.0
                if -V_REST < vi - v0 < V_REST:
                    vi = v0
                if vi >= vth:
                    vi = vr
                    ai += adapt_b[i]               # adaptation: every spike "tires" the neuron
                    ref_until[i] = t + ref_steps
                    counts[i] += 1
                    for p in range(indptr[i], indptr[i + 1]):  # will arrive in D steps
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
                    act[i] = 0                     # back at rest: leaves the list
                else:
                    alist[w] = i
                    w += 1
            nact = w
            t += 1
        nact_box[0] = nact
        return t

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"

# Parameters from Shiu et al. 2024 (times in ms, potentials in mV)
DEFAULT_PARAMS = dict(
    v0=-52.0,      # resting potential
    vr=-52.0,      # reset potential after a spike
    vth=-45.0,     # spike threshold
    tau_m=20.0,    # membrane time constant
    tau_s=5.0,     # synaptic time constant
    t_ref=2.2,     # refractory period
    delay=1.8,     # synaptic delay
    w_syn=0.275,   # weight of a single synapse
    f_poi=250.0,   # weight multiplier for Poisson stimuli
    # Adaptation (not in Shiu's model, off by default): every spike adds
    # adapt_b mV of "fatigue" that decays with time constant tau_a. It prevents
    # activity from sustaining itself forever.
    adapt_b=0.0,
    tau_a=200.0,
    # Synapses between Kenyon cells (mushroom body). The connectome has many of them, but in
    # the real fly the KCs stay quiet; in the model they keep exciting each other forever.
    # False = we cut them (only KC -> KC, the rest of the mushroom body stays).
    kc_recurrent=True,
    # Antennal lobe local interneurons (lLN1_bc, il3LN6, ...) that the connectome
    # assigns a modulator (dopamine, serotonin, octopamine): the estimate is unreliable
    # and in the real fly they are mostly GABAergic. True = we make them inhibitory.
    fix_ln_modulatory=False,
    # Neuromodulators (octopamine, dopamine, serotonin) act slowly through
    # metabotropic receptors, not as fast excitation. Shiu's model treats them
    # as fast excitatory. "keep" = as in Shiu; "oa" = only the octopaminergic neurons
    # by name (OA-...) give no fast excitation; "all" = no neuromodulator
    # (by name or by estimate) gives fast excitation, except the Kenyon cells
    # (estimated dopamine but cholinergic) and the interneurons already corrected.
    modulators="keep",
)

# Recommended configuration, chosen with the diagnostic scripts (07-13):
# Shiu's model with three connectome errors corrected (antennal lobe
# interneurons, KC->KC synapses, neuromodulators treated as fast excitation) plus
# light adaptation. Escape and taste (sugar -> MN9) as in the original, stable
# olfaction: the brain calms down when the odor goes away.
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

        self.D = max(1, round(self.p["delay"] / dt))       # delay steps
        self.ref_steps = round(self.p["t_ref"] / dt)
        self.decay_s = np.float32(np.exp(-dt / self.p["tau_s"]))
        self.decay_a = np.float32(np.exp(-dt / self.p["tau_a"]))
        # per-neuron adaptation (0 = exempt, see exempt_adaptation)
        self.adapt_b = np.full(self.N, self.p["adapt_b"], np.float32)
        self.k_m = np.float32(dt / self.p["tau_m"])
        self.poi_w = np.float32(self.p["w_syn"] * self.p["f_poi"])

        self._ct = self.neurons["cell_type"].fillna("").astype(str).str.upper()

        self.kc_cut = 0
        if not self.p["kc_recurrent"]:
            kc = self._ct.str.startswith("KC").to_numpy()
            col = np.repeat(np.arange(self.N), np.diff(self.indptr))   # presynaptic neuron of each synapse
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
            self.data[sel] = -np.abs(self.data[sel])                       # from excitatory to inhibitory
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
            sel = mod[col] & (self.data > 0)                  # remove only the fast excitation
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

    # ------------------------------------------------------------------ state
    def sibling(self, seed=None):
        """Another brain with the same connectome (weights and neuron table shared,
        no copy in memory) but with its own state: used to run several flies."""
        b = copy.copy(self)
        b.rng = np.random.default_rng(seed)
        b.reset()                                    # potentials, currents, delays: all new
        return b

    def reset(self):
        """Brings the brain back to rest. The background noise stays set."""
        N = self.N
        self.v = np.full(N, self.p["v0"], np.float32)
        self.x = np.zeros(N, np.float32)                   # synaptic current
        self.a = np.zeros(N, np.float32)                   # adaptation
        self.ref_until = np.zeros(N, np.int64)
        self.ring = np.zeros((self.D, N), np.float32)      # delay buffer
        # Numba engine structures (see _run_numba)
        self.mark = np.zeros((self.D, N), np.uint8)
        self.tlist = np.zeros((self.D, N), np.int32)
        self.tcount = np.zeros(self.D, np.int64)
        self.act = np.zeros(N, np.uint8)
        self.alist = np.zeros(N, np.int32)
        self.nact = np.zeros(1, np.int64)
        self.t = 0
        self.counts = np.zeros(N, np.int32)
        self.clear_stimulus()

    # -------------------------------------------------------------- lookup
    def _pick(self, mask, side=None):
        if side:
            mask = mask & (self._side == side)
        return self._idx[mask.to_numpy()]

    def find(self, cell_type, side=None):
        """Neurons of an exact cell type (optional: 'left'/'right')."""
        t = cell_type.upper()
        return self._pick((self._ct == t) | (self._hb == t), side)

    def find_prefix(self, prefix, side=None):
        """Neurons whose type starts with a prefix (e.g. 'ORN_')."""
        return self._pick(self._ct.str.startswith(prefix.upper()), side)

    def find_superclass(self, super_class, side=None):
        sc = self.neurons["super_class"].fillna("").astype(str).str.lower()
        return self._pick(sc == super_class.lower(), side)

    def types_with_prefix(self, prefix):
        """Distinct cell types starting with a prefix, with their neuron counts."""
        names = self.neurons["cell_type"].fillna("").astype(str)
        hit = names[names.str.upper().str.startswith(prefix.upper())]
        return hit.value_counts().sort_index()

    def describe(self, i):
        r = self.neurons.iloc[i]
        ct = r["cell_type"] if isinstance(r["cell_type"], str) else "?"
        return f"{ct} ({r['side']}, {r['super_class']})"

    def exempt_adaptation(self, idx):
        """These neurons never tire (e.g. the escape circuit)."""
        self.adapt_b[np.asarray(idx, np.int64)] = 0.0

    # ------------------------------------------------------------ input
    def clear_stimulus(self):
        self.stim_idx = np.array([], np.int64)
        self.stim_p = np.array([], np.float32)

    def set_stimulus(self, idx, rate_hz):
        """Poisson stimulus on a group of neurons (replaces the current one).
        rate_hz can be a single number or an array, one per neuron."""
        idx = np.asarray(idx, np.int64)
        rates = np.broadcast_to(np.asarray(rate_hz, np.float32), idx.shape)
        keep = rates > 0
        self.stim_idx = idx[keep]
        self.stim_p = (rates[keep] * self.dt * 1e-3).astype(np.float32)

    def add_stimulus(self, idx, rate_hz):
        """Adds a stimulus to the current one (e.g. a neuron clicked in the explorer)."""
        idx = np.asarray(idx, np.int64)
        if len(idx) == 0 or rate_hz <= 0:
            return
        p = np.full(len(idx), rate_hz * self.dt * 1e-3, np.float32)
        self.stim_idx = np.concatenate([self.stim_idx, idx])
        self.stim_p = np.concatenate([self.stim_p, p])

    def set_background(self, idx, rate_hz):
        """Spontaneous activity: each neuron of the group fires on average rate_hz times per second."""
        self.bg_idx = np.asarray(idx, np.int64)
        self.bg_p = float(rate_hz) * self.dt * 1e-3

    # ---------------------------------------------------------- simulation
    def step(self):
        p = self.p
        slot = self.t % self.D

        # incoming synaptic input (spikes from D steps ago)
        self.x += self.ring[slot]
        self.ring[slot] = 0.0

        # external stimuli
        if len(self.stim_idx):
            hit = self.rng.random(len(self.stim_idx)) < self.stim_p
            if hit.any():
                self.x[self.stim_idx[hit]] += self.poi_w

        # background noise: how many neurons "fire" in this step, then which ones
        nb = len(self.bg_idx)
        if nb and self.bg_p > 0:
            n = self.rng.binomial(nb, self.bg_p)
            if n:
                self.x[self.bg_idx[self.rng.integers(0, nb, n)]] += self.poi_w

        # membrane (frozen during the refractory period)
        dv = self.k_m * (self.x - (self.v - p["v0"]) - self.a)
        dv[self.ref_until > self.t] = 0.0
        self.v += dv
        self.x *= self.decay_s
        self.a *= self.decay_a

        # spikes
        spk = np.flatnonzero(self.v >= p["vth"])
        if len(spk):
            self.v[spk] = p["vr"]
            self.a[spk] += self.adapt_b[spk]
            self.ref_until[spk] = self.t + self.ref_steps
            self.counts[spk] += 1
            self._deliver(spk, self.ring[slot])         # will arrive in D steps

        self.t += 1
        return spk

    def _deliver(self, spk, ring):
        if len(spk) <= 4:                               # few spikes: simple path
            for j in spk:
                s, e = self.indptr[j], self.indptr[j + 1]
                ring[self.indices[s:e]] += self.data[s:e]
            return
        # many spikes: gather all the columns in one go
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
