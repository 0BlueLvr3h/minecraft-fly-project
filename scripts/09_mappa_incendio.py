r"""
Mappa dell'incendio olfattivo
------------------------------
Accende UN glomerulo alla volta (entrambe le antenne, intensità bassa) con la
configurazione del server, e per ognuno risponde a tre domande:

  1. si incendia?           spike al secondo, decolli spuri
  2. chi lo innesca?        i tipi di neuroni più attivi nell'istante in cui parte
  3. dove brucia?           classi e tipi più attivi durante tutto l'incendio

Controlla anche l'APL (il neurone che dovrebbe tenere silenziose le cellule di
Kenyon): se spara al massimo vuol dire che è saturo e non riesce a frenarle.

Uso:
  python scripts\09_mappa_incendio.py
  python scripts\09_mappa_incendio.py --hz 10 --sec 1.5
"""

import argparse

import numpy as np

import circuits
from circuits import FlyCircuits
from flysim import RECOMMENDED, FlyBrain

WINDOW_MS = 50.0
IGNITION = 1000          # spike in una finestra da 50 ms (= 20.000/s): incendio


def top_types(brain, counts, exclude, n=8):
    c = counts.copy()
    c[exclude] = 0
    df = brain.neurons.assign(s=c)
    t = df.groupby(df["cell_type"].fillna("?"))["s"].sum().sort_values(ascending=False)
    t = t[t > 0].head(n)
    return ", ".join(f"{k} {v:,}" for k, v in t.items()) or "(nessuno)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hz", type=float, default=20.0, help="frequenza sugli ORN del glomerulo")
    ap.add_argument("--sec", type=float, default=1.0, help="secondi per glomerulo")
    args = ap.parse_args()

    brain = FlyBrain(dt=0.2, seed=0, params=RECOMMENDED)
    circ = FlyCircuits(brain)
    gf = np.concatenate([circ.gf["L"], circ.gf["R"]])
    apl = brain.find("APL")
    print(f"Configurazione del server (interneuroni corretti: {brain.ln_fixed}). APL trovati: {len(apl)}. "
          f"Stimolo: {args.hz:.0f} Hz per {args.sec} s su un glomerulo alla volta.")

    glomeruli = [g for types in circuits.ODORS.values() for g in types]
    for g in glomeruli:
        idx = brain.find(g)
        if len(idx) == 0:
            print(f"\n=== {g}: non trovato")
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
        status = f"SI, dopo {onset * WINDOW_MS:.0f} ms" if onset is not None else "no"
        print(f"\n=== {g}  ({len(idx)} ORN)   incendio: {status}   "
              f"{sps:,.0f} spike/s   decolli spuri (spike GF): {gf_spikes}")
        print("    spike ogni 50 ms: " + " ".join(f"{int(s)}" for s in per_win[:12])
              + (" ..." if len(per_win) > 12 else ""))

        if onset is not None:
            start = max(0, onset - 1)
            lead = sum(wins[start:onset + 1])
            print(f"    chi lo innesca: {top_types(brain, lead, idx)}")
            print(f"    dove brucia:    {top_types(brain, total, idx)}")
            df = brain.neurons.assign(s=total)
            cls = df.groupby(df["super_class"].fillna("?"))["s"].sum().sort_values(ascending=False).head(5)
            print("    per classe:     " + ", ".join(f"{k} {v:,}" for k, v in cls.items()))
        else:
            print(f"    più attivi:     {top_types(brain, total, idx)}")

        if len(apl):
            print(f"    APL: {total[apl].sum() / len(apl) / args.sec:.0f} Hz "
                  f"(sopra ~350 Hz è praticamente saturo)")

    print("\nCon questo scegliamo dove intervenire (APL graduale, controllo del guadagno, ...).")


if __name__ == "__main__":
    main()
