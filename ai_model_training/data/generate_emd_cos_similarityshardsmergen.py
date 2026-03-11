#!/usr/bin/env python3
# merge_and_align_shards_to_json.py

import os, re, json, argparse, orjson, gzip
from typing import Iterable, List, Dict, Any

EXP_ESF, EXP_NORM = 640, 64      # exakt wie im Dataset
FEATURES = 44

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --------- Helpers: CV-Key, Längencheck, Fold-/Shard-Sortierung ---------

def parse_key(k: str):
    p = k.split("_")
    return "_".join(p[:-3]), p[-3], p[-2], int(p[-1])

def ok_len(store: Dict[str, Any], cls, inst, perc, idx, L: int) -> bool:
    try:
        v = store[cls][str(inst)][str(perc)][idx]
        return isinstance(v, list) and len(v) == L
    except Exception:
        return False

def pair_is_ok(p: Dict[str, Any], esf, norm) -> bool:
    cr, ir, pr, xr = parse_key(p["esf_ref"])
    cs, is_, ps, xs = parse_key(p["esf_scan"])
    return (ok_len(esf,  cr, ir, pr, xr, EXP_ESF) and
            ok_len(esf,  cs, is_, ps, xs, EXP_ESF) and
            ok_len(norm, cr, ir, pr, xr, EXP_NORM) and
            ok_len(norm, cs, is_, ps, xs, EXP_NORM))

def natural_fold_key(k: str):
    m = re.match(r"fold(\d+)$", k)
    return int(m.group(1)) if m else k

def shard_sort_key(p: str):
    m = re.search(r"\.part(\d+)\.jsonl$", p)
    return int(m.group(1)) if m else p

def _resolve_shard_path(in_dir: str, p: str) -> str:
    if os.path.isabs(p) and os.path.exists(p):
        return p
    if os.path.exists(p):
        return p
    return os.path.join(in_dir, os.path.basename(p))

def iter_shard_lines(in_dir: str, shards: List[str]) -> Iterable[bytes]:
    for sp in sorted(shards, key=shard_sort_key):
        sp = _resolve_shard_path(in_dir, sp)
        if not os.path.exists(sp):
            continue
        with open(sp, "rb") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line

# --------- Kernfunktion: Merge + Align in EIN JSON-Dict ---------

def merge_and_align_shards_to_json(
    in_dir: str,
    cv_path: str,
    esf_path: str,
    normal_path: str,
    out_path: str,
    placeholder: str = "zeros",     # "zeros" oder "null"
    folds: List[str] = None,
    check_len: bool = True
):
    # Metadaten und Rohdaten laden
    with open(os.path.join(in_dir, "index.json"), "r", encoding="utf-8") as f:
        idx_meta = json.load(f)
    with open(cv_path, "r", encoding="utf-8") as f:
        cv_all = json.load(f)
    with open(esf_path, "rb") as f:
        esf = orjson.loads(f.read())
    with open(normal_path, "rb") as f:
        norm = orjson.loads(f.read())

    folds_sorted = folds if folds else sorted(cv_all.keys(), key=natural_fold_key)
    use_gzip = out_path.endswith(".gz")
    out_fh = gzip.open(out_path, "wb") if use_gzip else open(out_path, "wb")

    ph_null  = b"null"
    ph_zeros = orjson.dumps([0.0] * FEATURES)
    use_null = (placeholder == "null")

    def write_split(pairs: List[Dict[str, Any]], shards: List[str], tag: str):
        it = iter_shard_lines(in_dir, shards) if shards else iter(())
        consumed = 0
        skipped = 0
        written = 0

        out_fh.write(b"[")
        first = True
        for p in pairs:
            if pair_is_ok(p, esf, norm):
                try:
                    raw = next(it)
                except StopIteration:
                    raw = None
                if raw is not None and check_len:
                    try:
                        arr = orjson.loads(raw)
                        if not (isinstance(arr, list) and len(arr) == FEATURES):
                            raw = None
                        else:
                            raw = orjson.dumps(arr)  # normalisieren
                    except Exception:
                        raw = None
                val = raw if raw is not None else (ph_null if use_null else ph_zeros)
                if raw is not None:
                    consumed += 1
            else:
                val = ph_null if use_null else ph_zeros
                skipped += 1

            if not first:
                out_fh.write(b",")
            out_fh.write(val)
            first = False
            written += 1
        out_fh.write(b"]")

        # Rest prüfen
        leftover = sum(1 for _ in it)
        if leftover:
            print(f"[WARN] {tag}: {leftover} ungenutzte Feature-Zeilen.")
        print(f"✓ {tag}: pairs={len(pairs)} | ok→{consumed} | not-ok→{skipped} | written={written}")

    # Streaming-JSON schreiben
    out_fh.write(b"{")
    first_fold = True
    for fold in folds_sorted:
        if fold not in cv_all:
            print(f"[WARN] {fold} nicht in CV-Datei. Skip.")
            continue
        if not first_fold:
            out_fh.write(b",")
        first_fold = False

        out_fh.write(orjson.dumps(fold))
        out_fh.write(b":{")

        meta = idx_meta.get(fold, {})
        cv_f = cv_all[fold]

        for j, split in enumerate(["train", "val", "test"]):
            if j > 0:
                out_fh.write(b",")
            out_fh.write(orjson.dumps(split))
            out_fh.write(b":")

            pairs = cv_f.get(split, [])
            shards = meta.get(split, {}).get("shards", [])
            write_split(pairs, shards, f"{fold}-{split}")

        out_fh.write(b"}")
    out_fh.write(b"}")
    out_fh.close()
    print(f"🎉 Fertig: {out_path}")

# --------- CLI ---------

def _main2():
    ap = argparse.ArgumentParser(
        description="Shards zu EINER JSON-Datei mergen und an cv6 ausrichten."
    )
    # Defaults: alles im Skript-Ordner, Shards in ./features_stream
    ap.add_argument("--in_dir", default=os.path.join(SCRIPT_DIR, "features_stream"),
                    help="features_stream (mit index.json)")
    ap.add_argument("--cv", default=os.path.join(SCRIPT_DIR, "cv6_info.json"),
                    help="cv6_info.json")
    ap.add_argument("--esf", default=os.path.join(SCRIPT_DIR, "verf_esf_dataset_2_instances_merged.json"),
                    help="verf_esf_dataset_2_instances_merged.json")
    ap.add_argument("--normal", default=os.path.join(SCRIPT_DIR, "verf_normal_xray_hist_dataset_2_instances_merged.json"),
                    help="verf_normal_xray_hist_dataset_2_instances_merged.json")
    ap.add_argument("--out", default=os.path.join(SCRIPT_DIR, "features_all_folds_aligned.json"),
                    help="Ausgabe: .json oder .json.gz")
    ap.add_argument("--placeholder", choices=["zeros", "null"], default="zeros",
                    help="Platzhalter für gedroppte Paare")
    ap.add_argument("--folds", nargs="*", help="Optional: nur bestimmte Folds, z.B. fold0 fold1")
    ap.add_argument("--no_check", action="store_true", help="Feature-Länge nicht prüfen (schneller)")
    args = ap.parse_args()

    merge_and_align_shards_to_json(
        in_dir=args.in_dir,
        cv_path=args.cv,
        esf_path=args.esf,
        normal_path=args.normal,
        out_path=args.out,
        placeholder=args.placeholder,
        folds=args.folds,
        check_len=(not args.no_check)
    )

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Shards (JSONL)  ->  .npy pro Fold/Split (float16, memmap-freundlich).
- Input:  SCRIPT_DIR/features_stream/index.json + *.partXXX.jsonl
- Output: SCRIPT_DIR/features_memmap/foldX_train.npy, _val.npy, _test.npy
- Nur erfolgreiche Paare (keine Platzhalter). Perfekt für Dataset-memmap.
"""

import os
import re
import json
import argparse
import orjson
import numpy as np
from typing import Iterable, List


import os, re, json, argparse, orjson, numpy as np
from typing import Iterable, List
from numpy.lib.format import open_memmap

FEATURES = 44
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------- helpers ----------
def shard_sort_key(p: str):
    m = re.search(r"\.part(\d+)\.jsonl$", p)
    return int(m.group(1)) if m else p

def resolve_path(in_dir: str, p: str) -> str:
    if os.path.isabs(p): return p
    if os.path.exists(p): return p
    return os.path.join(in_dir, os.path.basename(p))

def iter_lines(in_dir: str, shards: List[str]) -> Iterable[bytes]:
    for sp in sorted(shards, key=shard_sort_key):
        sp = resolve_path(in_dir, sp)
        if not os.path.exists(sp):
            print(f"[WARN] Shard fehlt: {sp}")
            continue
        with open(sp, "rb") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line

def is_valid44(line: bytes) -> bool:
    if line == b"null":
        return False
    try:
        arr = orjson.loads(line)
    except Exception:
        return False
    return isinstance(arr, list) and (len(arr) == FEATURES)

def count_valid_rows(in_dir: str, shards: List[str]) -> int:
    n = 0
    for line in iter_lines(in_dir, shards):
        if is_valid44(line):
            n += 1
    return n

# ---------- writer ----------
def write_split(in_dir: str, shards: List[str], out_path: str) -> int:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if not shards:
        np.save(out_path, np.zeros((0, FEATURES), dtype=np.float16))
        print(f"  -> leer (0 Zeilen)")
        return 0

    # Pass 1: valide Zeilen zählen
    N = count_valid_rows(in_dir, shards)
    # Echte .npy-Datei mit Header + Memmap öffnen
    mm = open_memmap(out_path, mode="w+", dtype=np.float16, shape=(N, FEATURES))

    # Pass 2: schreiben
    k = 0
    bad = 0
    for line in iter_lines(in_dir, shards):
        if not is_valid44(line):
            bad += 1
            continue
        arr = orjson.loads(line)
        mm[k, :] = np.asarray(arr, dtype=np.float16)
        k += 1

    # Sicherheitscheck
    assert k == N, f"geschrieben {k} != gezählt {N}"
    del mm  # Datei sauber schließen
    print(f"  -> gültig: {k} | verworfen: {bad}")
    return k

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(
        description="Shards -> .npy pro Fold/Split (float16, memmap, korrekter Header)."
    )
    ap.add_argument("--in_dir", default=os.path.join(SCRIPT_DIR, "features_stream"),
                    help="Ordner mit index.json und *.jsonl Shards")
    ap.add_argument("--out_dir", default=os.path.join(SCRIPT_DIR, "features_memmap"),
                    help="Zielordner für .npy")
    ap.add_argument("--folds", nargs="*", default=None,
                    help="Optional: nur bestimmte Folds (z.B. fold4 fold5)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    index_path = os.path.join(args.in_dir, "index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"index.json nicht gefunden: {index_path}")

    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    folds = args.folds if args.folds else sorted(index.keys())
    for fold in folds:
        if fold not in index:
            print(f"[WARN] {fold} nicht in index.json. Skip.")
            continue
        meta = index[fold]
        for split in ["train", "val", "test"]:
            shards = meta.get(split, {}).get("shards", [])
            out_path = os.path.join(args.out_dir, f"{fold}_{split}.npy")
            print(f"✓ {fold}-{split}: schreibe {out_path}")
            n = write_split(args.in_dir, shards, out_path)
            print(f"✓ {fold}-{split}: {n} Zeilen → {out_path}")

if __name__ == "__main__":
    main()