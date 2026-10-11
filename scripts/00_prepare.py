"""
Fase 0 - Preparazione dati FlyWire v783
----------------------------------------
Legge i 3 file scaricati, controlla che siano coerenti e produce:
  data/processed/neurons.parquet   -> anagrafica neuroni (indice, root_id, tipo, nt, posizione)
  data/processed/weights.npz       -> matrice pesi sparsa CSR [post x pre], con segno
  data/processed/report.txt        -> riepilogo + neuroni chiave trovati

Uso (dalla cartella mosca\):
  python scripts\00_prepare.py
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow.feather as feather
import pyarrow.ipc as ipc
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

F_IDS = RAW / "proofread_root_ids_783.npy"
F_CONN = RAW / "proofread_connections_783.feather"
F_ANN = RAW / "Supplemental_file1_neuron_annotations.tsv"

# Segno del neurotrasmettitore (convenzione Shiu et al. 2024):
# ACh eccitatorio, GABA e glutammato inibitori. I modulatori li trattiamo
# come eccitatori per ora: agiscono lentamente e nel modello LIF pesano poco.
NT_SIGN = {
    "acetylcholine": 1, "gaba": -1, "glutamate": -1,
    "dopamine": 1, "serotonin": 1, "octopamine": 1,
}

# Neuroni che ci servono per l'MVP della fuga (e qualcuno per dopo)
KEY_TYPES = ["LC4", "LPLC2", "DNp01", "DNa02", "DNg02", "MDN", "P9"]

log_lines = []
def log(msg=""):
    print(msg)
    log_lines.append(str(msg))


def check_files():
    missing = [f.name for f in (F_IDS, F_CONN, F_ANN) if not f.exists()]
    if missing:
        sys.exit(f"File mancanti in {RAW}: {missing}")


def load_ids():
    ids = np.load(F_IDS).astype(np.int64)
    log(f"Root ID validati: {len(ids):,}")
    return ids


def load_connections():
    # Leggiamo solo lo schema prima, per non caricare colonne inutili
    with open(F_CONN, "rb") as fh:
        cols = ipc.open_file(fh).schema.names
    log(f"Colonne connessioni: {cols}")

    wanted = [c for c in ("pre_pt_root_id", "post_pt_root_id", "syn_count",
                          "neuropil") if c in cols]
    if not {"pre_pt_root_id", "post_pt_root_id", "syn_count"} <= set(wanted):
        sys.exit("Il file connessioni non ha le colonne attese: controlla i nomi sopra.")

    conn = feather.read_table(F_CONN, columns=wanted).to_pandas()
    log(f"Righe connessioni (per neuropilo): {len(conn):,}")

    # Una coppia pre->post può comparire in più neuropili: sommiamo
    conn = (conn.groupby(["pre_pt_root_id", "post_pt_root_id"], sort=False)
                ["syn_count"].sum().reset_index())
    log(f"Coppie pre->post uniche: {len(conn):,}")
    log(f"Sinapsi totali: {conn['syn_count'].sum():,}")
    return conn


def load_annotations():
    ann = pd.read_csv(F_ANN, sep="\t", low_memory=False)
    ann["root_id"] = ann["root_id"].astype(np.int64)
    log(f"Neuroni annotati: {len(ann):,}")
    return ann


def build(ids, conn, ann):
    # Indice compatto 0..N-1 per ogni neurone
    idx_of = pd.Series(np.arange(len(ids)), index=ids)

    neurons = pd.DataFrame({"idx": np.arange(len(ids)), "root_id": ids})
    keep = [c for c in ("cell_type", "hemibrain_type", "super_class", "cell_class",
                        "side", "top_nt", "soma_x", "soma_y", "soma_z",
                        "pos_x", "pos_y", "pos_z") if c in ann.columns]
    neurons = neurons.merge(ann[["root_id"] + keep], on="root_id", how="left")

    no_ann = neurons["super_class"].isna().sum() if "super_class" in neurons else -1
    log(f"Neuroni senza annotazione: {no_ann:,}")

    # Posizione per l'esploratore: soma se c'è, altrimenti punto di ancoraggio.
    # Coordinate in voxel 4x4x40 nm -> convertiamo in micrometri.
    for a, scale in (("x", 4), ("y", 4), ("z", 40)):
        src = neurons.get(f"soma_{a}")
        alt = neurons.get(f"pos_{a}")
        v = src.fillna(alt) if src is not None else alt
        neurons[a] = v * scale / 1000.0

    # Segno per neurone presinaptico
    nt = neurons["top_nt"].fillna("").str.lower()
    neurons["sign"] = nt.map(NT_SIGN).fillna(1).astype(np.int8)
    log("Neurotrasmettitori:")
    log(nt.replace("", "(sconosciuto)").value_counts().to_string())

    # Teniamo solo connessioni tra neuroni validati
    m = conn["pre_pt_root_id"].isin(idx_of.index) & conn["post_pt_root_id"].isin(idx_of.index)
    log(f"Connessioni scartate (neuroni non validati): {(~m).sum():,}")
    conn = conn[m]

    pre = idx_of[conn["pre_pt_root_id"].values].values
    post = idx_of[conn["post_pt_root_id"].values].values
    w = conn["syn_count"].values.astype(np.float32) * neurons["sign"].values[pre]

    n = len(ids)
    W = sp.csr_matrix((w, (post, pre)), shape=(n, n), dtype=np.float32)
    log(f"Matrice pesi: {n:,} x {n:,}, {W.nnz:,} connessioni, "
        f"{W.data.nbytes / 1e6:.0f} MB di dati")
    return neurons, W


def find_key_neurons(neurons):
    log("\nNeuroni chiave:")
    ct = neurons.get("cell_type", pd.Series(dtype=str)).fillna("").astype(str)
    hb = neurons.get("hemibrain_type", pd.Series(dtype=str)).fillna("").astype(str)
    for t in KEY_TYPES:
        hit = neurons[(ct.str.upper() == t.upper()) | (hb.str.upper() == t.upper())]
        sides = hit["side"].value_counts().to_dict() if "side" in hit else {}
        log(f"  {t:7s} -> {len(hit):4d} neuroni  {sides}")
    gf = neurons[ct.str.contains("giant", case=False) | hb.str.contains("giant", case=False)]
    if len(gf):
        log(f"  (trovati anche {len(gf)} neuroni con 'giant' nel nome)")


def main():
    check_files()
    ids = load_ids()
    conn = load_connections()
    ann = load_annotations()
    neurons, W = build(ids, conn, ann)
    find_key_neurons(neurons)

    neurons.to_parquet(OUT / "neurons.parquet", index=False)
    sp.save_npz(OUT / "weights.npz", W)
    (OUT / "report.txt").write_text("\n".join(log_lines), encoding="utf-8")
    log(f"\nFatto. Output in {OUT}")


if __name__ == "__main__":
    main()