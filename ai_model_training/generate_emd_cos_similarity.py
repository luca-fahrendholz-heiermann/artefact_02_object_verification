import json
import orjson
import numpy as np
from tqdm import tqdm
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import cosine as cosine_distance
import os

# ---------------------------------------------------
# Normierungen
def normalize_hist_local(hist):
    max_val = np.max(np.abs(hist))
    return hist / max_val if max_val > 1e-8 else np.zeros_like(hist)

def normalize_hist_global(hist_block):
    max_val = np.max(np.abs(hist_block))
    return hist_block / max_val if max_val > 1e-8 else np.zeros_like(hist_block)

# ---------------------------------------------------
# Feature-Berechnungen
def compute_emd(desc1, desc2):
    out = []
    # lokal
    for i in range(10):
        h1 = normalize_hist_local(desc1[i*64:(i+1)*64])
        h2 = normalize_hist_local(desc2[i*64:(i+1)*64])
        try:
            out.append(wasserstein_distance(h1, h2) / 64.0)
        except:
            out.append(1.0)
    # global
    g1 = normalize_hist_global(desc1.flatten())
    g2 = normalize_hist_global(desc2.flatten())
    for i in range(10):
        h1 = g1[i*64:(i+1)*64]
        h2 = g2[i*64:(i+1)*64]
        try:
            out.append(wasserstein_distance(h1, h2) / 64.0)
        except:
            out.append(1.0)
    return out

def compute_cos(desc1, desc2):
    out = []
    for i in range(10):
        h1 = normalize_hist_local(desc1[i*64:(i+1)*64])
        h2 = normalize_hist_local(desc2[i*64:(i+1)*64])
        if np.linalg.norm(h1)==0 and np.linalg.norm(h2)==0:
            out.append(0.0)
        else:
            try:
                out.append(np.clip(cosine_distance(h1,h2),0,1))
            except:
                out.append(1.0)
    g1 = normalize_hist_global(desc1.flatten())
    g2 = normalize_hist_global(desc2.flatten())
    for i in range(10):
        h1 = g1[i*64:(i+1)*64]
        h2 = g2[i*64:(i+1)*64]
        if np.linalg.norm(h1)==0 and np.linalg.norm(h2)==0:
            out.append(0.0)
        else:
            try:
                out.append(np.clip(cosine_distance(h1,h2),0,1))
            except:
                out.append(1.0)
    return out

# ---------------------------------------------------
# Key Parser
def parse_key(key_str):
    parts = key_str.split("_")
    cls = "_".join(parts[:-3])
    inst = parts[-3]
    perc = parts[-2]
    idx = int(parts[-1])
    return cls, inst, perc, idx

# ---------------------------------------------------
# Lade ESF-Daten
def load_esf(path):
    with open(path,"rb") as f:
        return orjson.loads(f.read())

def get_vec(esf_data, cls, inst, perc, idx):
    try:
        vec = esf_data[cls][str(inst)][str(perc)][idx]
        return np.array(vec, dtype=np.float32)
    except Exception as e:
        print(f"[WARN] Missing {cls}/{inst}/{perc}/{idx} -> {e}")
        return None

# ---------------------------------------------------
# Haupt-Funktion
def process_fold(esf_data, fold_name, fold_dict, out_dir):
    for split in ["train","val","test"]:
        out_path = f"{out_dir}/{fold_name}_{split}_features.jsonl"
        with open(out_path,"wb") as fout:
            for pair in tqdm(fold_dict[split], desc=f"{fold_name}-{split}"):
                cls_r, inst_r, perc_r, idx_r = parse_key(pair["esf_ref"])
                cls_s, inst_s, perc_s, idx_s = parse_key(pair["esf_scan"])
                d1 = get_vec(esf_data, cls_r, inst_r, perc_r, idx_r)
                d2 = get_vec(esf_data, cls_s, inst_s, perc_s, idx_s)
                if d1 is None or d2 is None:
                    continue
                feats = compute_emd(d1,d2)+compute_cos(d1,d2) # 40 Werte
                fout.write(orjson.dumps(feats)+b"\n")
        print(f"✅ Saved {split} → {out_path}")

# ---------------------------------------------------
# Main
if __name__=="__main__":
    esf_json = "verf_esf_dataset_2.json"
    cv_info = "cv6_info.json"
    out_dir = "features_out"
    os.makedirs(out_dir, exist_ok=True)

    esf_data = load_esf(esf_json)
    with open(cv_info,"r") as f:
        cv = json.load(f)

    for fold_name, fold_dict in cv.items():
        process_fold(esf_data, fold_name, fold_dict, out_dir)
