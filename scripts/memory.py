r"""
memory - il corpo fungiforme che impara
----------------------------------------
Il connettoma è una fotografia: le sinapsi hanno una forza fissa. Qui aggiungiamo
l'unica plasticità che serve per l'apprendimento associativo della mosca, quella
misurata negli esperimenti (Hige et al. 2015, Cohn et al. 2015, Berry et al. 2018):

  - le cellule di Kenyon (KC) codificano l'odore: ogni odore ne accende poche, sempre
    le stesse
  - i dopaminergici (DAN) dicono "buono" (PAM) o "cattivo" (PPL1), ognuno nel suo
    compartimento del corpo fungiforme
  - se una KC è attiva mentre arriva la dopamina nel compartimento, la sua sinapsi sul
    neurone di uscita (MBON) di quel compartimento si indebolisce
  - dopamina senza la KC attiva la riporta piano verso il valore originale
    (si può disimparare), e col tempo si dimentica

Quale DAN modula quale MBON non lo scriviamo a mano: lo dice il connettoma
(sinapsi DAN -> MBON). Lo stesso per quali MBON leggere: quelli che ricevono
quasi tutto il loro input dalle KC.

VALENZA. I MBON dei compartimenti PAM spingono ad allontanarsi, quelli dei
compartimenti PPL1 ad avvicinarsi (Aso et al. 2014). Dopo una ricompensa l'odore
spegne i primi, dopo una punizione i secondi. La valenza imparata dell'odore che la
mosca sente è quindi

    valenza = (ingresso KC ai MBON PPL1, adesso / con i pesi originali)
            - (ingresso KC ai MBON PAM,  adesso / con i pesi originali)

0 = odore mai associato a niente, > 0 = buono, < 0 = cattivo. Leggiamo l'ingresso
delle KC e non le scariche dei MBON perché nel modello i MBON sono dominati da una
rete del corno laterale che si accende a scatti (vedi 15_corpo_fungiforme.py).

Le vie che portano zucchero e dolore ai DAN non funzionano nel modello (sono lente
o passano dal cordone nervoso): reward() e punish() danno i Hz da mettere sui DAN,
come negli esperimenti di optogenetica.
"""

import math

import numpy as np
import scipy.sparse as sp

from flysim import PROC

# dopaminergici del rinforzo (accesi dal plugin al posto di zucchero e dolore)
REWARD_DANS = ["PAM01", "PAM02", "PAM05", "PAM06", "PAM07", "PAM08", "PAM11"]   # gamma5, beta'2, gamma4, alpha1
PUNISH_DANS = ["PPL101", "PPL103", "PPL106"]                                     # gamma1pedc, gamma2alpha'1, alpha3
DAN_RATE = 150.0          # Hz sui dopaminergici con rinforzo pieno

KC_MIN_FRAC = 0.7         # MBON letti: almeno il 70% del loro input dalle KC
TAU_KC = 1.0              # s: traccia di attività delle KC (quanto "ricorda" l'odore appena sentito)
TAU_DAN = 0.5             # s: traccia della dopamina
KC_ON = 3.0               # traccia KC (circa spike/s) sotto cui la KC non conta: impara solo
                          # chi risponde forte, le KC più specifiche dell'odore (meno confusione)
KC_FULL = 8.0             # traccia KC che vale "pienamente attiva"
DAN_MIN = 25.0            # Hz di dopamina (media pesata per sinapsi) sotto cui non è rinforzo:
                          # i DAN hanno un'attività di fondo e qualche odore li accende un po'
DAN_FULL = 75.0           # Hz di dopamina che valgono "rinforzo pieno"
LEARN_RATE = 0.8          # per secondo: KC e dopamina al massimo -> sinapsi a metà in meno di 1 s
RECOVER_RATE = 0.02       # per secondo: dopamina senza KC -> la sinapsi torna verso l'originale
FORGET_S = 1800.0         # s: col tempo si dimentica (mezz'ora)
MIN_INPUT = 0.05          # sotto questa frazione dell'ingresso tipico: odore troppo debole, valenza 0


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

        # anatomia dal connettoma grezzo (prima delle correzioni del simulatore)
        W = sp.load_npz(PROC / "weights.npz").tocsr()
        A = abs(W[self.mbon])
        is_kc = np.zeros(b.N, bool)
        is_kc[self.kc] = True
        tot = np.maximum(1, np.asarray(A.sum(1)).ravel())
        kc_frac = np.asarray(A[:, is_kc].sum(1)).ravel() / tot
        S = A[:, self.dan].toarray()                               # MBON x DAN, sinapsi
        self.dan_w = (S / np.maximum(1, S.sum(1, keepdims=True))).astype(np.float32)
        is_pam = np.isin(self.dan, pam)
        comp = np.where(S[:, is_pam].sum(1) > S[:, ~is_pam].sum(1), 1, -1)   # 1 = PAM, -1 = PPL1
        comp[S.sum(1) < 20] = 0
        self.comp = comp * (kc_frac >= KC_MIN_FRAC)                # 0 = non letto

        # sinapsi KC -> MBON nei pesi del simulatore (CSC: colonna = presinaptico)
        where = np.full(b.N, -1, np.int64)
        where[self.mbon] = np.arange(len(self.mbon))
        cols = np.repeat(np.arange(b.N), np.diff(b.indptr))
        sel = is_kc[cols] & (where[b.indices] >= 0)
        self.pos = np.flatnonzero(sel)
        self.pre = cols[self.pos]                                   # indice KC (nel cervello)
        self.post = where[b.indices[self.pos]]                      # indice MBON (0..len(mbon))
        self.kc_of = np.searchsorted(self.kc, self.pre)             # indice KC (0..len(kc))
        self.w0 = b.data[self.pos].astype(np.float32).copy()
        self.reset()

    def sibling(self, brain):
        """Lo stesso corpo fungiforme per un'altra mosca, con la sua memoria (vuota).
        La mosca riceve una copia privata dei pesi, perché imparare li cambia."""
        m = object.__new__(MushroomBody)
        m.__dict__.update(self.__dict__)
        m.brain = brain
        brain.data = brain.data.copy()
        m.reset()
        return m

    def reset(self):
        """Dimentica tutto."""
        self.w = self.w0.copy()
        self.brain.data[self.pos] = self.w
        self.kc_trace = np.zeros(len(self.kc), np.float32)
        self.dan_trace = np.zeros(len(self.mbon), np.float32)
        self.valence = 0.0
        self.familiar = 0.0
        self._typ = None

    # ------------------------------------------------------------ rinforzo
    def reinforcement(self, reward=0.0, punish=0.0):
        """Stimoli da aggiungere: [(indici, Hz), ...] per i DAN del rinforzo."""
        out = []
        if reward > 0:
            out.append((self.reward_idx, min(1.0, reward) * DAN_RATE))
        if punish > 0:
            out.append((self.punish_idx, min(1.0, punish) * DAN_RATE))
        return out

    # ------------------------------------------------------------ apprendimento
    def update(self, delta, window_ms):
        """Dopo ogni finestra: aggiorna le tracce, impara, legge la valenza."""
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
        w -= (LEARN_RATE * dt * e_syn * d_syn) * w                          # KC + dopamina: indebolisce
        w += (RECOVER_RATE * dt * (1 - e_syn) * d_syn) * (self.w0 - w)      # solo dopamina: recupera
        w += (dt / FORGET_S) * (self.w0 - w)                                 # col tempo: dimentica
        self.brain.data[self.pos] = w
        self._read(elig)

    def _read(self, elig):
        """Valenza imparata dell'odore sentito adesso (vedi in cima al file)."""
        k = elig[self.kc_of] ** 2                      # pesa di più le KC che rispondono forte (più specifiche)
        now = np.bincount(self.post, weights=k * self.w, minlength=len(self.mbon))
        orig = np.bincount(self.post, weights=k * self.w0, minlength=len(self.mbon))
        if self._typ is None:                                    # ingresso tipico di un odore pieno
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
        """Quanto è cambiata la memoria in tutto (0 = niente imparato)."""
        return float(1 - (self.w.sum() / max(self.w0.sum(), 1e-9)))
