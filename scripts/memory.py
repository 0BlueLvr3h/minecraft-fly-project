r"""
memory - the mushroom body that learns
--------------------------------------
The connectome is a snapshot: synapses have a fixed strength. Here we add
the only plasticity needed for the fly's associative learning, the one
measured in experiments (Hige et al. 2015, Cohn et al. 2015, Berry et al. 2018):

  - Kenyon cells (KCs) encode the odor: each odor activates a few of them, always
    the same ones
  - dopaminergic neurons (DANs) say "good" (PAM) or "bad" (PPL1), each in its own
    compartment of the mushroom body
  - if a KC is active while dopamine arrives in the compartment, its synapse onto
    the output neuron (MBON) of that compartment weakens
  - dopamine without the KC being active slowly brings it back towards the original value
    (it can be unlearned), and over time it is forgotten

Which DAN modulates which MBON is not written by hand: the connectome tells us
(DAN -> MBON synapses). The same goes for which MBONs to read: those that receive
almost all of their input from the KCs.

VALENCE. The MBONs of the PAM compartments drive avoidance, those of the
PPL1 compartments drive approach (Aso et al. 2014). After a reward the odor
silences the former, after a punishment the latter. The learned valence of the odor the
fly is smelling is therefore

    valence = (KC input to the PPL1 MBONs, now / with the original weights)
            - (KC input to the PAM MBONs,  now / with the original weights)

0 = odor never associated with anything, > 0 = good, < 0 = bad. We read the KC
input and not the MBON firing because in the model the MBONs are dominated by a
lateral horn network that switches on in bursts (see 15_mushroom_body.py).

The pathways that carry sugar and pain to the DANs do not work in the model (they are slow
or go through the ventral nerve cord): reward() and punish() give the Hz to put on the DANs,
as in optogenetics experiments.
"""

import math

import numpy as np
import scipy.sparse as sp

from flysim import PROC

# reinforcement dopaminergic neurons (driven by the plugin in place of sugar and pain)
REWARD_DANS = ["PAM01", "PAM02", "PAM05", "PAM06", "PAM07", "PAM08", "PAM11"]   # gamma5, beta'2, gamma4, alpha1
PUNISH_DANS = ["PPL101", "PPL103", "PPL106"]                                     # gamma1pedc, gamma2alpha'1, alpha3
DAN_RATE = 150.0          # Hz on the dopaminergic neurons with full reinforcement

KC_MIN_FRAC = 0.7         # MBONs read: at least 70% of their input from KCs
TAU_KC = 1.0              # s: KC activity trace (how long it "remembers" the odor just smelled)
TAU_DAN = 0.5             # s: dopamine trace
KC_ON = 3.0               # KC trace (about spikes/s) below which the KC does not count: only strongly
                          # responding KCs learn, the most odor-specific ones (less confusion)
KC_FULL = 8.0             # KC trace that counts as "fully active"
DAN_MIN = 25.0            # dopamine Hz (synapse-weighted mean) below which it is not reinforcement:
                          # DANs have background activity and some odors activate them a little
DAN_FULL = 75.0           # dopamine Hz that counts as "full reinforcement"
LEARN_RATE = 0.8          # per second: KC and dopamine at maximum -> synapse halved in under 1 s
RECOVER_RATE = 0.02       # per second: dopamine without KC -> the synapse returns towards the original
FORGET_S = 1800.0         # s: forgetting over time (half an hour)
MIN_INPUT = 0.05          # below this fraction of the typical input: odor too weak, valence 0


class MushroomBody:
    def __init__(self, brain):
        b = self.brain = brain
        self.kc = b.find_prefix("KC")
        self.mbon = b.find_prefix("MBON")
        pam = b.find_prefix("PAM")
        ppl1 = b.find_prefix("PPL1")
        self.dan = np.concatenate([pam, ppl1])
        self.reward_idx = np.concatenate([b.find(t) for t in REWARD_DANS])
        self.punish_idx = np.concatenate([b.find(t) for t in PUNISH_DANS])

        # anatomy from the raw connectome (before the simulator's corrections)
        W = sp.load_npz(PROC / "weights.npz").tocsr()
        A = abs(W[self.mbon])
        is_kc = np.zeros(b.N, bool)
        is_kc[self.kc] = True
        tot = np.maximum(1, np.asarray(A.sum(1)).ravel())
        kc_frac = np.asarray(A[:, is_kc].sum(1)).ravel() / tot
        S = A[:, self.dan].toarray()                               # MBON x DAN, synapses
        self.dan_w = (S / np.maximum(1, S.sum(1, keepdims=True))).astype(np.float32)
        is_pam = np.isin(self.dan, pam)
        comp = np.where(S[:, is_pam].sum(1) > S[:, ~is_pam].sum(1), 1, -1)   # 1 = PAM, -1 = PPL1
        comp[S.sum(1) < 20] = 0
        self.comp = comp * (kc_frac >= KC_MIN_FRAC)                # 0 = not read

        # KC -> MBON synapses in the simulator weights (CSC: column = presynaptic)
        where = np.full(b.N, -1, np.int64)
        where[self.mbon] = np.arange(len(self.mbon))
        cols = np.repeat(np.arange(b.N), np.diff(b.indptr))
        sel = is_kc[cols] & (where[b.indices] >= 0)
        self.pos = np.flatnonzero(sel)
        self.pre = cols[self.pos]                                   # KC index (in the brain)
        self.post = where[b.indices[self.pos]]                      # MBON index (0..len(mbon))
        self.kc_of = np.searchsorted(self.kc, self.pre)             # KC index (0..len(kc))
        self.w0 = b.data[self.pos].astype(np.float32).copy()
        self.reset()

    def sibling(self, brain):
        """The same mushroom body for another fly, with its own (empty) memory.
        The fly gets a private copy of the weights, because learning changes them."""
        m = object.__new__(MushroomBody)
        m.__dict__.update(self.__dict__)
        m.brain = brain
        brain.data = brain.data.copy()
        m.reset()
        return m

    def reset(self):
        """Forgets everything."""
        self.w = self.w0.copy()
        self.brain.data[self.pos] = self.w
        self.kc_trace = np.zeros(len(self.kc), np.float32)
        self.dan_trace = np.zeros(len(self.mbon), np.float32)
        self.valence = 0.0
        self.familiar = 0.0
        self._typ = None

    # ------------------------------------------------------------ reinforcement
    def reinforcement(self, reward=0.0, punish=0.0):
        """Stimuli to add: [(indices, Hz), ...] for the reinforcement DANs."""
        out = []
        if reward > 0:
            out.append((self.reward_idx, min(1.0, reward) * DAN_RATE))
        if punish > 0:
            out.append((self.punish_idx, min(1.0, punish) * DAN_RATE))
        return out

    # ------------------------------------------------------------ learning
    def update(self, delta, window_ms):
        """After each window: updates the traces, learns, reads the valence."""
        dt = window_ms / 1000.0
        kc_spk = delta[self.kc].astype(np.float32)
        self.kc_trace = self.kc_trace * math.exp(-dt / TAU_KC) + kc_spk / TAU_KC
        dan_hz = delta[self.dan].astype(np.float32) / dt
        dop = self.dan_w @ dan_hz
        a = math.exp(-dt / TAU_DAN)
        self.dan_trace = self.dan_trace * a + dop * (1 - a)

        elig = np.clip((self.kc_trace - KC_ON) / (KC_FULL - KC_ON), 0, 1)
        d = np.clip((self.dan_trace - DAN_MIN) / (DAN_FULL - DAN_MIN), 0, 1)
        e_syn = elig[self.kc_of]
        d_syn = d[self.post]
        w = self.w
        w -= (LEARN_RATE * dt * e_syn * d_syn) * w                          # KC + dopamine: weakens
        w += (RECOVER_RATE * dt * (1 - e_syn) * d_syn) * (self.w0 - w)      # dopamine only: recovers
        w += (dt / FORGET_S) * (self.w0 - w)                                 # over time: forgets
        self.brain.data[self.pos] = w
        self._read(elig)

    def _read(self, elig):
        """Learned valence of the odor being smelled now (see the top of the file)."""
        k = elig[self.kc_of] ** 2                      # weighs strongly responding KCs more (more specific)
        now = np.bincount(self.post, weights=k * self.w, minlength=len(self.mbon))
        orig = np.bincount(self.post, weights=k * self.w0, minlength=len(self.mbon))
        if self._typ is None:                                    # typical input of a full odor
            self._typ = float(np.bincount(self.post, weights=self.w0, minlength=len(self.mbon))[
                self.comp != 0].sum()) * 0.03
        pam, ppl = self.comp == 1, self.comp == -1
        o_pam, o_ppl = orig[pam].sum(), orig[ppl].sum()
        self.familiar = float(min(1.0, (o_pam + o_ppl) / max(self._typ, 1e-9)))
        if self.familiar < MIN_INPUT or o_pam <= 0 or o_ppl <= 0:
            self.valence = 0.0
            return
        self.valence = float(now[ppl].sum() / o_ppl - now[pam].sum() / o_pam)

    def strength(self):
        """How much the memory has changed overall (0 = nothing learned)."""
        return float(1 - (self.w.sum() / max(self.w0.sum(), 1e-9)))
