r"""
Odori dei player: quali sa distinguere il cervello?
----------------------------------------------------
La mosca riconosce un player solo dall'odore, e lo impara nelle cellule di Kenyon (KC).
Perché ogni player resti "lui", il suo odore deve:
  - accendere abbastanza KC (sennò la memoria ha poco su cui scrivere)
  - accendere sempre le stesse (sennò ogni volta sembra un odore nuovo)
  - sovrapporsi poco con gli odori degli altri player (sennò li confonde)
  - NON accendere il Giant Fiber: per una mosca l'odore di un corpo non è un pericolo
    innato (alcuni mix di glomeruli arrivano alla fuga e la farebbero scappare da sola)
  - costare poco al cervello: con più player vicini gli odori si sommano, e un odore
    che accende mezzo cervello rallenta la simulazione sotto il tempo reale

Prima proviamo i glomeruli uno per uno e scartiamo quelli "inefficienti": quelli che
accendono molto il resto del cervello e poco il corpo fungiforme. Poi generiamo tanti odori candidati (un gruppo di glomeruli, ognuno con la sua intensità),
li facciamo annusare al cervello due volte a circa 2 blocchi di distanza e teniamo i
migliori. Il cervello non viene toccato: scegliamo solo gli odori.

Scrive data/processed/player_odors.json, che il server usa per dare un odore a ogni
player. Da rifare solo se cambia il connettoma o la configurazione del cervello.

Uso:
  python scripts\17_odori_player.py              (prova gli odori e sceglie: qualche minuto)
  python scripts\17_odori_player.py --riscegli   (sceglie di nuovo tra quelli già provati: secondi)
"""

import json
import sys
import time

import numpy as np

from circuits import ODOR_RATE, ODORS, FlyCircuits, player_rate
from flysim import PROC, RECOMMENDED, FlyBrain

CANDIDATES = 300
KEEP = 16                 # quanti odori diversi (oltre: due player possono condividerne uno)
CONC = 0.6                # circa un player a 2 blocchi
MS = 1000.0
MAX_COST = 40000          # spike/s di tutto il cervello per un odore a 2 blocchi (oltre: troppo pesante)
MAX_OVERLAP = 0.12        # sovrapposizione massima tra le KC di due odori scelti
MIN_KC = 40               # KC affidabili minime perché un odore valga
# Glomeruli esclusi: già usati per frutta e marcio, o con un significato innato forte:
# feromoni (DA1, VA1d, VA1v, DL3) e ammoniaca/ammine (VM1, VC5, VM6m, VM6v, VM6l), che nel
# connettoma arrivano al Giant Fiber: da soli fanno scappare la mosca.
RESERVED = ({g for gl in ODORS.values() for g in gl}
            | {"ORN_DA1", "ORN_VA1d", "ORN_VA1v", "ORN_DL3"}
            | {"ORN_VM1", "ORN_VC5", "ORN_VM6m", "ORN_VM6v", "ORN_VM6l"})
# Glomerulo inefficiente: da solo costa più di GLOM_HEAVY spike/s, oppure più di GLOM_COST
# accendendo meno di GLOM_MIN_KC cellule di Kenyon.
GLOM_HEAVY = 30000
GLOM_COST = 10000
GLOM_MIN_KC = 40
OUT = PROC / "player_odors.json"
CACHE = PROC / "player_odor_candidates.json"      # odori già provati, per --riscegli


def simulate():
    """Fa annusare al cervello i glomeruli e poi gli odori candidati; salva tutto in CACHE."""
    t0 = time.perf_counter()
    brain = FlyBrain(dt=0.2, seed=17, params=RECOMMENDED)
    circ = FlyCircuits(brain)
    gf = np.concatenate([circ.gf["L"], circ.gf["R"]])
    kc = brain.find_prefix("KC")
    ct = brain.neurons["cell_type"].fillna("").astype(str)
    pool = sorted(t for t in ct[ct.str.startswith("ORN_")].unique() if t not in RESERVED)
    print(f"Cervello pronto in {time.perf_counter() - t0:.1f} s; provo {len(pool)} glomeruli uno per uno")
    keep = []
    for g in pool:
        cost, n_kc = 0.0, 0
        for _ in range(2):                       # due prove: il cervello è rumoroso
            brain.reset()
            brain.set_stimulus(brain.find(g), player_rate(CONC))
            brain.run(MS)
            cost += brain.counts.sum() / (MS / 1000) / 2
            n_kc += int((brain.counts[kc] >= 2).sum()) / 2
        bad = cost > GLOM_HEAVY or (cost > GLOM_COST and n_kc < GLOM_MIN_KC)
        if bad:
            print(f"  scarto {g[4:]:5s}: {cost / 1000:.0f}k spike/s per {n_kc:.0f} KC")
        else:
            keep.append(g)
    pool = keep
    print(f"{len(pool)} glomeruli usabili, {CANDIDATES} odori da provare")

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
    """Sceglie il gruppo più grande di odori che si sovrappongono poco tra loro."""
    for c in cands:
        c["kc"] = set(c["kc"])
    n_gf = sum(c["gf"] > 0 for c in cands)
    n_cost = sum(c["cost"] > MAX_COST for c in cands)
    good = [c for c in cands if len(c["kc"]) >= MIN_KC and c["gf"] == 0 and c["cost"] <= MAX_COST]
    print(f"\nScartati: {n_gf} accendono il Giant Fiber, {n_cost} troppo pesanti (> {MAX_COST} spike/s)")
    print(f"Odori buoni (almeno {MIN_KC} KC affidabili, niente fuga, leggeri): {len(good)} su {len(cands)}")

    def overlap(a, b):
        return len(a["kc"] & b["kc"]) / len(a["kc"] | b["kc"])

    # conflitti: coppie di odori che si sovrappongono troppo (la mosca li confonderebbe)
    conflicts = [{j for j, b in enumerate(good) if j != i and overlap(a, b) > MAX_OVERLAP}
                 for i, a in enumerate(good)]
    score = [len(c["kc"]) * c["reliability"] for c in good]
    left, picked = set(range(len(good))), []
    while left and len(picked) < KEEP:
        # prima l'odore che esclude meno altri; a parità, quello che la mosca riconosce meglio
        i = min(left, key=lambda k: (len(conflicts[k] & left), -score[k]))
        picked.append(i)
        left -= conflicts[i] | {i}
    chosen = [good[i] for i in picked]
    # ordine: i primi player ricevono gli odori più diversi tra loro (le coppie più simili
    # entrano in gioco solo quando ci sono tanti player)
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

    print(f"Scelti {len(chosen)} odori:")
    for i, c in enumerate(chosen):
        print(f"  {i:2d}: {len(c['glomeruli']):2d} glomeruli, KC affidabili {len(c['kc']):4d} "
              f"({100 * len(c['kc']) / n_kc:.1f}%), ripetibilità {c['reliability']:.2f}, "
              f"sovrapposizione max {c['overlap']:.2f}, costo {c['cost'] / 1000:.0f}k spike/s")
    if len(chosen) < KEEP:
        print(f"Nota: {len(chosen)} odori abbastanza distinti, i player oltre li condivideranno")
    return chosen


def main():
    t0 = time.perf_counter()
    if "--riscegli" in sys.argv:
        if not CACHE.exists():
            raise SystemExit("Nessun odore già provato: lancia lo script senza --riscegli")
        data = json.loads(CACHE.read_text(encoding="utf-8"))
        cands, n_kc = data["candidates"], data["n_kc"]
        print(f"Riscelgo tra {len(cands)} odori già provati")
    else:
        cands, n_kc = simulate()
    chosen = select(cands, n_kc)

    OUT.write_text(json.dumps([{"glomeruli": c["glomeruli"], "profile": c["profile"]} for c in chosen],
                              indent=1), encoding="utf-8")
    assign = PROC / "player_assign.json"          # gli odori sono cambiati: si riassegnano da capo
    if assign.exists():
        assign.unlink()
        print("Assegnazioni dei player azzerate (ognuno riceverà uno dei nuovi odori)")
    print(f"\nSalvato {OUT} in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
