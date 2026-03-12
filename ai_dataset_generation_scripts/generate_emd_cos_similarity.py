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
    for i in range(11):
        h1 = normalize_hist_local(desc1[i*64:(i+1)*64])
        h2 = normalize_hist_local(desc2[i*64:(i+1)*64])
        try:
            out.append(wasserstein_distance(h1, h2) / 64.0)
        except:
            out.append(1.0)
    # global
    g1 = normalize_hist_global(desc1.flatten())
    g2 = normalize_hist_global(desc2.flatten())
    for i in range(11):
        h1 = g1[i*64:(i+1)*64]
        h2 = g2[i*64:(i+1)*64]
        try:
            out.append(wasserstein_distance(h1, h2) / 64.0)
        except:
            out.append(1.0)
    return out

def compute_cos(desc1, desc2):
    out = []
    for i in range(11):
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
    for i in range(11):
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
# ---------------------------------------------------
# Haupt-Funktion
def process_all_old(esf_data, cv, out_json):
    #os.makedirs(os.path.dirname(out_json), exist_ok=True)

    # Falls schon existiert, laden (um fortzusetzen)
    if os.path.exists(out_json):
        with open(out_json, "rb") as f:
            all_results = orjson.loads(f.read())
    else:
        all_results = {}

    for fold_name, fold_dict in cv.items():
        fold_out = {}
        for split in ["train", "val", "test"]:
            feats_list = []
            for pair in tqdm(fold_dict[split], desc=f"{fold_name}-{split}"):
                cls_r, inst_r, perc_r, idx_r = parse_key(pair["esf_ref"])
                cls_s, inst_s, perc_s, idx_s = parse_key(pair["esf_scan"])
                d1 = get_vec(esf_data, cls_r, inst_r, perc_r, idx_r)
                d2 = get_vec(esf_data, cls_s, inst_s, perc_s, idx_s)
                if d1 is None or d2 is None:
                    continue
                feats_np = compute_emd(d1, d2) + compute_cos(d1, d2)
                feats_clean = [round(np.float16(x), 4) for x in feats_np]
                # Rundung auf 4 Nachkommastellen und Konvertierung zu float

                feats_list.append(feats_clean)

            fold_out[split] = feats_list

        # Update in Haupt-Dict
        all_results[fold_name] = fold_out

        # 🔥 Nach jedem Fold abspeichern
        with open(out_json, "wb") as f:
            f.write(orjson.dumps(all_results, option=orjson.OPT_SERIALIZE_NUMPY))
        print(f"✅ Fold {fold_name} gespeichert → {out_json}")

    print("✅ Alle Folds fertig und gespeichert.")

def process_all(esf_data, normalxray_data, cv, out_json):
    if os.path.exists(out_json):
        with open(out_json, "rb") as f:
            all_results = orjson.loads(f.read())
    else:
        all_results = {}

    for fold_name, fold_dict in cv.items():
        fold_out = {}
        for split in ["train", "val", "test"]:
            feats_list = []
            for pair in tqdm(fold_dict[split], desc=f"{fold_name}-{split}"):
                cls_r, inst_r, perc_r, idx_r = parse_key(pair["esf_ref"])
                cls_s, inst_s, perc_s, idx_s = parse_key(pair["esf_scan"])

                # Hole beide Vektoren und kombiniere sie
                esf_vec_r = get_vec(esf_data, cls_r, inst_r, perc_r, idx_r)
                esf_vec_s = get_vec(esf_data, cls_s, inst_s, perc_s, idx_s)

                norm_vec_r = get_vec(normalxray_data, cls_r, inst_r, perc_r, idx_r)
                norm_vec_s = get_vec(normalxray_data, cls_s, inst_s, perc_s, idx_s)

                if esf_vec_r is None or esf_vec_s is None or norm_vec_r is None or norm_vec_s is None:
                    print('Error: Ref: ',cls_r, inst_r, perc_r, idx_r  )
                    print('Scan: ',cls_s, inst_s, perc_s, idx_s )
                    continue

                # Kombinieren → 704-dimensional
                d1 = np.concatenate([esf_vec_r, norm_vec_r]).astype(np.float32)
                d2 = np.concatenate([esf_vec_s, norm_vec_s]).astype(np.float32)
                # Ähnlichkeitsmaße berechnen
                feats_np = compute_emd(d1, d2) + compute_cos(d1, d2)
                feats_clean = [round(np.float16(x), 4) for x in feats_np]

                feats_list.append(feats_clean)

            fold_out[split] = feats_list

        all_results[fold_name] = fold_out

        with open(out_json, "wb") as f:
            f.write(orjson.dumps(all_results, option=orjson.OPT_SERIALIZE_NUMPY))
        print(f"✅ Fold {fold_name} gespeichert → {out_json}")

    print("✅ Alle Folds fertig und gespeichert.")

# ---------------------------------------------------
# Main
if __name__ == "__main__":
    esf_json = "verf_esf_dataset_2_instances_merged.json"
    normal_json = "verf_normal_xray_hist_dataset_2_instances_merged.json"
    cv_info_path = "cv6_info.json"
    out_json = "features_all_folds_merged.json"
    esf_data = load_esf(esf_json)
    normalxray_data = load_esf(normal_json)
    with open(cv_info_path, "r") as f:
        cv = json.load(f)

    features_dict = process_all(esf_data, normalxray_data, cv, out_json)