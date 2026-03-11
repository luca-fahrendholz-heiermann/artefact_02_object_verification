#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Feature-Extraktion mit Checkpoints.
- JSONL-Shards pro Split mit Rotation.
- Atomare State-Datei pro Split (Resume).
- Konstante Featurelänge: 44 (11 Blöcke × 2 Metriken × lokal+global).
"""

import os
import json
import argparse
import orjson
import numpy as np
from tqdm import tqdm
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import cosine as cosine_distance

# ----------------------------- Konstanten
TARGET_DIM = 704      # 11 * 64
BLOCK = 64
N_BLOCKS = TARGET_DIM // BLOCK        # 11
FEATURES_PER_PAIR = 4 * N_BLOCKS      # 44

# Default-IO-Parameter
DEFAULT_FLUSH_EVERY = 10_000
DEFAULT_SHARD_SIZE  = 200_000

# ----------------------------- Normalisierungen
def normalize_hist_local(hist: np.ndarray) -> np.ndarray:
    hist = np.asarray(hist, dtype=np.float32)
    max_val = float(np.max(np.abs(hist))) if hist.size else 0.0
    if max_val > 1e-8:
        return hist / max_val
    return np.zeros_like(hist, dtype=np.float32)

def normalize_hist_global(hist_block: np.ndarray) -> np.ndarray:
    hist_block = np.asarray(hist_block, dtype=np.float32)
    max_val = float(np.max(np.abs(hist_block))) if hist_block.size else 0.0
    if max_val > 1e-8:
        return hist_block / max_val
    return np.zeros_like(hist_block, dtype=np.float32)

# ----------------------------- Utility
def pad_to_len(vec: np.ndarray, L: int = TARGET_DIM) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32).ravel()
    if vec.size >= L:
        return vec[:L]
    out = np.zeros(L, dtype=np.float32)
    if vec.size:
        out[:vec.size] = vec
    return out

def parse_key(key_str: str):
    parts = key_str.split("_")
    cls = "_".join(parts[:-3])
    inst = parts[-3]
    perc = parts[-2]
    idx = int(parts[-1])
    return cls, inst, perc, idx

def load_esf(path: str):
    with open(path, "rb") as f:
        return orjson.loads(f.read())

def get_vec(esf_data, cls, inst, perc, idx):
    try:
        lst = esf_data[cls][str(inst)][str(perc)]
        if not isinstance(lst, list):
            raise TypeError("Leaf is not list-like")
        if idx < 0 or idx >= len(lst):
            raise IndexError(f"idx {idx} out of range (len={len(lst)})")
        vec = lst[idx]
        return np.array(vec, dtype=np.float32)
    except Exception as e:
        print(f"[WARN] Missing {cls}/{inst}/{perc}/{idx} -> {e}")
        return None

# ----------------------------- Feature-Berechnung (fix 11 Blöcke)
def compute_emd_fixed(d1: np.ndarray, d2: np.ndarray):
    out = []
    # lokal
    for i in range(N_BLOCKS):
        h1 = normalize_hist_local(d1[i*BLOCK:(i+1)*BLOCK])
        h2 = normalize_hist_local(d2[i*BLOCK:(i+1)*BLOCK])
        try:
            out.append(float(wasserstein_distance(h1, h2)) / BLOCK)
        except Exception:
            out.append(1.0)
    # global
    g1 = normalize_hist_global(d1)
    g2 = normalize_hist_global(d2)
    for i in range(N_BLOCKS):
        h1 = g1[i*BLOCK:(i+1)*BLOCK]
        h2 = g2[i*BLOCK:(i+1)*BLOCK]
        try:
            out.append(float(wasserstein_distance(h1, h2)) / BLOCK)
        except Exception:
            out.append(1.0)
    return out  # 22

def compute_cos_fixed(d1: np.ndarray, d2: np.ndarray):
    out = []
    # lokal
    for i in range(N_BLOCKS):
        h1 = normalize_hist_local(d1[i*BLOCK:(i+1)*BLOCK])
        h2 = normalize_hist_local(d2[i*BLOCK:(i+1)*BLOCK])
        if (np.linalg.norm(h1) == 0.0) and (np.linalg.norm(h2) == 0.0):
            out.append(0.0)
        else:
            try:
                out.append(float(np.clip(cosine_distance(h1, h2), 0, 1)))
            except Exception:
                out.append(1.0)
    # global
    g1 = normalize_hist_global(d1)
    g2 = normalize_hist_global(d2)
    for i in range(N_BLOCKS):
        h1 = g1[i*BLOCK:(i+1)*BLOCK]
        h2 = g2[i*BLOCK:(i+1)*BLOCK]
        if (np.linalg.norm(h1) == 0.0) and (np.linalg.norm(h2) == 0.0):
            out.append(0.0)
        else:
            try:
                out.append(float(np.clip(cosine_distance(h1, h2), 0, 1)))
            except Exception:
                out.append(1.0)
    return out  # 22

def compute_feats(pair, esf_data, normalxray_data):
    # Keys
    cls_r, inst_r, perc_r, idx_r = parse_key(pair["esf_ref"])
    cls_s, inst_s, perc_s, idx_s = parse_key(pair["esf_scan"])
    # Laden
    esf_vec_r = get_vec(esf_data, cls_r, inst_r, perc_r, idx_r)
    esf_vec_s = get_vec(esf_data, cls_s, inst_s, perc_s, idx_s)
    norm_vec_r = get_vec(normalxray_data, cls_r, inst_r, perc_r, idx_r)
    norm_vec_s = get_vec(normalxray_data, cls_s, inst_s, perc_s, idx_s)
    if esf_vec_r is None or esf_vec_s is None or norm_vec_r is None or norm_vec_s is None:
        return None
    # Kombinieren
    d1 = np.concatenate([esf_vec_r, norm_vec_r]).astype(np.float32, copy=False)
    d2 = np.concatenate([esf_vec_s, norm_vec_s]).astype(np.float32, copy=False)
    # Fix-Länge
    d1 = pad_to_len(d1, TARGET_DIM)
    d2 = pad_to_len(d2, TARGET_DIM)
    # Features
    feats = compute_emd_fixed(d1, d2) + compute_cos_fixed(d1, d2)  # 44
    feats = [round(float(np.float16(x)), 4) for x in feats]
    return feats

# ----------------------------- Checkpoint-Helpers
def state_paths(out_dir, fold, split):
    base = os.path.join(out_dir, f"{fold}-{split}")
    return base + ".state.json", base + ".state.json.tmp"

def shard_path(out_dir, fold, split, shard_id):
    return os.path.join(out_dir, f"{fold}-{split}.part{shard_id:03d}.jsonl")

def load_state(out_dir, fold, split):
    path, _ = state_paths(out_dir, fold, split)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"idx": 0, "shard_id": 0, "in_shard": 0, "written_total": 0, "skipped_total": 0}

def save_state(out_dir, fold, split, st):
    path, tmp = state_paths(out_dir, fold, split)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f)
    os.replace(tmp, path)

# ----------------------------- Processing mit Checkpoints
def process_split_with_checkpoints(esf_data,
                                   normalxray_data,
                                   pairs,
                                   out_dir,
                                   fold,
                                   split,
                                   flush_every=DEFAULT_FLUSH_EVERY,
                                   shard_size=DEFAULT_SHARD_SIZE):
    os.makedirs(out_dir, exist_ok=True)
    st = load_state(out_dir, fold, split)

    # Öffne aktuellen Shard im Append-Modus.
    fp_path = shard_path(out_dir, fold, split, st["shard_id"])
    fp = open(fp_path, "ab")

    written_this_run = 0
    skipped_this_run = 0

    pbar = tqdm(range(st["idx"], len(pairs)), desc=f"{fold}-{split}", initial=st["idx"], total=len(pairs))
    for i in pbar:
        pair = pairs[i]
        feats = compute_feats(pair, esf_data, normalxray_data)
        if feats is None:
            st["skipped_total"] += 1
            skipped_this_run += 1
        else:
            fp.write(orjson.dumps(feats))
            fp.write(b"\n")
            st["in_shard"] += 1
            st["written_total"] += 1
            written_this_run += 1

        # Shard-Rotation
        if st["in_shard"] >= shard_size:
            fp.close()
            st["shard_id"] += 1
            st["in_shard"] = 0
            fp_path = shard_path(out_dir, fold, split, st["shard_id"])
            fp = open(fp_path, "ab")

        # Periodisches Flushen des State
        if ((i + 1) % flush_every) == 0:
            st["idx"] = i + 1
            save_state(out_dir, fold, split, st)

    # Final flush
    st["idx"] = len(pairs)
    save_state(out_dir, fold, split, st)
    fp.close()

    print(f"✅ {fold}-{split}: geschrieben={written_this_run}, übersprungen={skipped_this_run}, shard={st['shard_id']:03d}, in_shard={st['in_shard']}")
    return written_this_run, skipped_this_run

def process_all_with_checkpoints(esf_data,
                                 normalxray_data,
                                 cv,
                                 out_dir="features_stream",
                                 flush_every=DEFAULT_FLUSH_EVERY,
                                 shard_size=DEFAULT_SHARD_SIZE):
    os.makedirs(out_dir, exist_ok=True)
    index = {}
    total_written = 0
    total_skipped = 0

    for fold_name, fold_dict in cv.items():
        index[fold_name] = {}
        for split in ["train", "val", "test"]:
            pairs = fold_dict.get(split, [])
            w, s = process_split_with_checkpoints(
                esf_data, normalxray_data, pairs, out_dir, fold_name, split,
                flush_every=flush_every, shard_size=shard_size
            )
            total_written += w
            total_skipped += s
            # Liste vorhandener Shards erfassen
            shard_files = []
            shard_id = 0
            while True:
                p = shard_path(out_dir, fold_name, split, shard_id)
                if os.path.exists(p):
                    shard_files.append(p)
                    shard_id += 1
                else:
                    break
            # State lesen
            st = load_state(out_dir, fold_name, split)
            index[fold_name][split] = {
                "shards": shard_files,
                "state": st,
                "features_per_pair": FEATURES_PER_PAIR
            }

    # Index schreiben
    with open(os.path.join(out_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    print(f"🏁 Gesamt: geschrieben={total_written}, übersprungen={total_skipped}")
    print(f"📁 Index: {os.path.join(out_dir, 'index.json')}")
    return index

# ----------------------------- CLI / Main
def main():
    ap = argparse.ArgumentParser(description="Feature-Extraktion mit Checkpoints und Shards (JSONL).")
    ap.add_argument("--esf_json", default="verf_esf_dataset_3_instances_merged.json", help="Pfad zu ESF-JSON")
    ap.add_argument("--normal_json", default="verf_normal_xray_hist_dataset_3_instances_merged.json", help="Pfad zu Normal-XRay-JSON")
    ap.add_argument("--cv_info", default="cv6_info.json", help="Pfad zu CV-Info JSON")
    ap.add_argument("--out_dir", default="features_stream", help="Ausgabeverzeichnis")
    ap.add_argument("--flush_every", type=int, default=DEFAULT_FLUSH_EVERY, help="State-Flush alle N Paare")
    ap.add_argument("--shard_size", type=int, default=DEFAULT_SHARD_SIZE, help="Zeilen pro Shard-Datei")
    args = ap.parse_args()

    print("📦 Laden der Daten ...")
    esf_data = load_esf(args.esf_json)
    normalxray_data = load_esf(args.normal_json)
    with open(args.cv_info, "r", encoding="utf-8") as f:
        cv = json.load(f)

    print("🚀 Start Verarbeitung ...")
    process_all_with_checkpoints(
        esf_data=esf_data,
        normalxray_data=normalxray_data,
        cv=cv,
        out_dir=args.out_dir,
        flush_every=args.flush_every,
        shard_size=args.shard_size
    )

if __name__ == "__main__":
    main()