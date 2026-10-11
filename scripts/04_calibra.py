r"""
Calibrazione: camminata spontanea e olfatto
--------------------------------------------
Prima di collegare tutto a Minecraft, verifichiamo tre cose:

  1. i neuroni giusti esistono (camminata, sterzata, antenne)
  2. con un po' di rumore di fondo la mosca "cammina" da sola, ma senza
     decollare a caso
  3. un odore da un lato fa sterzare la mosca:
        frutta a sinistra   -> dovrebbe girare a SINISTRA (ci va incontro)
        marcio a sinistra   -> dovrebbe girare a DESTRA   (lo evita)

Uso (dalla cartella del progetto, con il venv attivo):
  python scripts\04_calibra.py
  python scripts\04_calibra.py --dur 3 --noise 0 2 5 10
"""

import argparse
import time

import numpy as np

from circuits import FlyCircuits
from flysim import FlyBrain

WINDOW_MS = 50.0


def simulate(brain, circ, ms, threats=None, odors=None):
    """Fa girare il cervello a finestre come il server, accumulando i risultati."""
    brain.reset()
    loom = circ.looming(threats)
    smell = circ.smell(odors)
    acc = {"fwd": 0.0, "back": 0.0, "turnL": 0.0, "turnR": 0.0}
    gf_spikes, takeoffs, n = 0, 0, 0
    t0 = time.perf_counter()
    for _ in range(int(ms / WINDOW_MS)):
        circ.apply(loom, smell)
        before = brain.counts.copy()
        brain.run(WINDOW_MS)
        out = circ.decode(brain.counts - before, WINDOW_MS)
        for k in acc:
            acc[k] += out["dn"][k]
        gf_spikes += sum(out["gf"])
        takeoffs += out["takeoff"]
        n += 1
    wall = time.perf_counter() - t0
    acc = {k: v / max(n, 1) for k, v in acc.items()}
    return acc, gf_spikes, takeoffs, (brain.counts > 0).sum(), ms / 1000 / wall


def steer(acc):
    tl, tr = acc["turnL"], acc["turnR"]
    if tl + tr == 0:
        return "nessuna sterzata"
    idx = (tl - tr) / (tl + tr)
    side = "SINISTRA" if idx > 0 else "DESTRA"
    return f"verso {side} (indice {idx:+.2f})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dt", type=float, default=0.2)
    ap.add_argument("--dur", type=float, default=2.0, help="secondi per prova")
    ap.add_argument("--noise", type=float, nargs="+", default=[0, 2, 5, 10],
                    help="frequenze di rumore di fondo da provare (Hz)")
    args = ap.parse_args()
    ms = args.dur * 1000

    t0 = time.perf_counter()
    brain = FlyBrain(dt=args.dt, seed=0)
    circ = FlyCircuits(brain)
    print(f"Cervello caricato in {time.perf_counter() - t0:.1f} s\n")

    # 1. neuroni trovati ------------------------------------------------
    print("=== 1. Neuroni trovati")
    lines, warnings = circ.report()
    for line in lines:
        print("   ", line)
    for w in warnings:
        print("    ATTENZIONE:", w)
    orn = brain.types_with_prefix("ORN")
    print(f"\n    Tipi ORN disponibili ({len(orn)}):")
    print("    " + ", ".join(f"{k}({v})" for k, v in orn.items()))
    for pre in ("DNp09", "DNa02", "MDN", "DNg02"):
        found = brain.types_with_prefix(pre)
        print(f"    Tipi che iniziano con {pre}: "
              + (", ".join(f"{k}({v})" for k, v in found.items()) or "nessuno"))

    # 2. rumore di fondo -------------------------------------------------
    print("\n=== 2. Attività spontanea (nessuno stimolo, solo rumore di fondo)")
    print("    rumore   avanti  indietro  sterz.L  sterz.R   GF  decolli  attivi   rt")
    for hz in args.noise:
        circ.set_noise(hz)
        acc, gf, tk, active, rt = simulate(brain, circ, ms)
        print(f"    {hz:5.1f} Hz {acc['fwd']:7.1f} {acc['back']:8.1f} {acc['turnL']:8.1f}"
              f" {acc['turnR']:8.1f} {gf:5d} {tk:7d} {active:8,d} {rt:5.2f}")
    circ.set_noise(0)

    # 3. odori ------------------------------------------------------------
    print("\n=== 3. Odori (senza rumore), sorgente a 1 blocco")
    print("    caso                     avanti  indietro  sterz.L  sterz.R   risultato")
    for name in circ.odor:
        for label, angle in (("sinistra", 90), ("destra", -90)):
            odors = [{"odor": name, "angle": angle, "dist": 1.0}]
            acc, gf, tk, active, rt = simulate(brain, circ, ms, odors=odors)
            case = f"{name} a {label}"
            print(f"    {case:24s} {acc['fwd']:6.1f} {acc['back']:9.1f} {acc['turnL']:8.1f}"
                  f" {acc['turnR']:8.1f}   {steer(acc)}")

    print("\nIncolla questo output in chat: da qui scegliamo rumore e soglie.")


if __name__ == "__main__":
    main()
