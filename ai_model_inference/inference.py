import argparse
import copy
import json
import logging
import os
import sys
from typing import Dict, Tuple

import numpy as np

try:
    import torch
except ImportError as exc:
    raise ImportError("PyTorch is required for inference.py. Please install torch in this environment.") from exc

import open3d as o3d
import laspy as lp

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "model")
if MODEL_DIR not in sys.path:
    sys.path.append(MODEL_DIR)

from ov_ai_model import MLPFlex
from ov_utils_mode import ChannelMaskWrapper, FeatMode
from preprocess_4_inference import (
    preprocess_pcd_for_esf,
    compute_esf_descriptor,
    generate_normal_xray_hist,
    features_for_object,
    normalize_to_minus_one_and_one_v2,
)


def read_pcd_in_any_format(path: str, sampling_points: int = 4096) -> o3d.geometry.PointCloud:
    ext = os.path.splitext(path)[1].lower()
    if ext in {".pcd", ".ply", ".xyz", ".xyzn", ".xyzrgb"}:
        pcd = o3d.io.read_point_cloud(path)
    elif ext in {".obj", ".stl", ".off", ".gltf", ".glb"}:
        mesh = o3d.io.read_triangle_mesh(path)
        pcd = mesh.sample_points_poisson_disk(sampling_points)
    elif ext == ".las":
        lasdata = lp.read(path)
        pts = np.column_stack([lasdata.x, lasdata.y, lasdata.z]).astype(np.float32)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
    elif ext in {".txt", ".asc", ".csv"}:
        pts = np.loadtxt(path, delimiter=",")
        if pts.ndim != 2 or pts.shape[1] < 3:
            raise ValueError(f"Unsupported text point cloud shape {pts.shape} for {path}")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts[:, :3])
    else:
        raise ValueError(f"Unsupported file extension '{ext}' for {path}")

    if len(pcd.points) == 0:
        raise ValueError(f"Loaded empty point cloud: {path}")
    return pcd


def _row_metrics(v1: np.ndarray, v2: np.ndarray) -> Tuple[float, float, float, float]:
    from scipy.spatial.distance import cosine
    from scipy.stats import wasserstein_distance

    v1 = np.asarray(v1, dtype=np.float32)
    v2 = np.asarray(v2, dtype=np.float32)

    cos_local = 1.0 - cosine(v1, v2)
    v1n = v1 / (np.linalg.norm(v1) + 1e-8)
    v2n = v2 / (np.linalg.norm(v2) + 1e-8)
    cos_global = 1.0 - cosine(v1n, v2n)
    emd_local = wasserstein_distance(v1, v2) / 64.0
    emd_global = wasserstein_distance(v1n, v2n) / 64.0

    if not np.isfinite(cos_local):
        cos_local = 0.0
    if not np.isfinite(cos_global):
        cos_global = 0.0
    if not np.isfinite(emd_local):
        emd_local = 1.0
    if not np.isfinite(emd_global):
        emd_global = 1.0
    return float(cos_local), float(cos_global), float(emd_local), float(emd_global)

from typing import Dict, Tuple, Literal
import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import wasserstein_distance


def _row_metrics_v1(v1: np.ndarray, v2: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Aktuelle Inference-Variante:
    [cos_local, cos_global, emd_local, emd_global]
    pro Block.
    """
    v1 = np.asarray(v1, dtype=np.float32)
    v2 = np.asarray(v2, dtype=np.float32)

    cos_local = 1.0 - cosine(v1, v2)

    v1n = v1 / (np.linalg.norm(v1) + 1e-8)
    v2n = v2 / (np.linalg.norm(v2) + 1e-8)

    cos_global = 1.0 - cosine(v1n, v2n)
    emd_local = wasserstein_distance(v1, v2) / 64.0
    emd_global = wasserstein_distance(v1n, v2n) / 64.0

    if not np.isfinite(cos_local):
        cos_local = 0.0
    if not np.isfinite(cos_global):
        cos_global = 0.0
    if not np.isfinite(emd_local):
        emd_local = 1.0
    if not np.isfinite(emd_global):
        emd_global = 1.0

    return float(cos_local), float(cos_global), float(emd_local), float(emd_global)


def _normalize_hist_local(h: np.ndarray) -> np.ndarray:
    """
    Lokale Max-Norm wie in der vorberechneten Extra-Feature-Pipeline.
    """
    h = np.asarray(h, dtype=np.float32).reshape(-1)
    m = float(np.max(np.abs(h))) if h.size > 0 else 0.0
    if m <= 1e-12:
        return np.zeros_like(h, dtype=np.float32)
    return (h / m).astype(np.float32)


def _normalize_hist_global(blocks: np.ndarray) -> np.ndarray:
    """
    Globale Max-Norm über alle 10 ESF-Blöcke bzw. über den 64D-Block.
    """
    blocks = np.asarray(blocks, dtype=np.float32)
    m = float(np.max(np.abs(blocks))) if blocks.size > 0 else 0.0
    if m <= 1e-12:
        return np.zeros_like(blocks, dtype=np.float32)
    return (blocks / m).astype(np.float32)


def _safe_cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine-DISTANZ, nicht Similarity.
    """
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)

    if np.allclose(a, 0.0) or np.allclose(b, 0.0):
        return 1.0

    d = float(cosine(a, b))
    if not np.isfinite(d):
        d = 1.0
    return float(np.clip(d, 0.0, 1.0))


def _safe_emd(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)

    d = float(wasserstein_distance(a, b))
    if not np.isfinite(d):
        d = 1.0
    return float(np.clip(d, 0.0, 1.0))


def _build_extra44_v2_precomputed_style(
    esf_target: np.ndarray,
    esf_source: np.ndarray,
    norm_target: np.ndarray,
    norm_source: np.ndarray,
) -> np.ndarray:
    """
    Variante 2:
    Rekonstruiert die Logik der vorberechneten Extra-Features.

    WICHTIG:
    Reihenfolge ist hier:
      [alle EMD-Werte, dann alle Cosine-Distanzen]
    und nicht blockweise [cos, cos, emd, emd].
    """

    esfA = np.asarray(esf_target, dtype=np.float32).reshape(10, 64)
    esfB = np.asarray(esf_source, dtype=np.float32).reshape(10, 64)

    # Lokale Max-Norm pro Block
    esfA_local = np.stack([_normalize_hist_local(row) for row in esfA], axis=0)
    esfB_local = np.stack([_normalize_hist_local(row) for row in esfB], axis=0)

    # Globale Max-Norm über alle 10x64
    esfA_global = _normalize_hist_global(esfA)
    esfB_global = _normalize_hist_global(esfB)

    # Norm-Hist ebenfalls lokal/global normalisieren
    normA_local = _normalize_hist_local(norm_target)
    normB_local = _normalize_hist_local(norm_source)

    normA_global = _normalize_hist_global(norm_target)
    normB_global = _normalize_hist_global(norm_source)

    emd_feats = []
    cos_feats = []

    # 10 ESF-Blöcke
    for i in range(10):
        emd_feats.append(_safe_emd(esfA_local[i], esfB_local[i]))
        cos_feats.append(_safe_cosine_distance(esfA_local[i], esfB_local[i]))

        emd_feats.append(_safe_emd(esfA_global[i], esfB_global[i]))
        cos_feats.append(_safe_cosine_distance(esfA_global[i], esfB_global[i]))

    # 1 Normal-Hist Block
    emd_feats.append(_safe_emd(normA_local, normB_local))
    cos_feats.append(_safe_cosine_distance(normA_local, normB_local))

    emd_feats.append(_safe_emd(normA_global, normB_global))
    cos_feats.append(_safe_cosine_distance(normA_global, normB_global))

    x44 = np.asarray(emd_feats + cos_feats, dtype=np.float32)

    if x44.size != 44:
        raise ValueError(f"Variant v2 produced {x44.size} extra features instead of 44.")

    return x44


def build_input_vectors(
    model_target_pcd: o3d.geometry.PointCloud,
    scan_source_pcd: o3d.geometry.PointCloud,
    esf_exe_path: str,
    legacy_normalize_main_diff: bool = False,
    extra_feature_variant: Literal["v1", "v2"] = "v1",
) -> Dict[str, np.ndarray]:

    target_esf_pcd = preprocess_pcd_for_esf(copy.deepcopy(model_target_pcd))
    source_esf_pcd = preprocess_pcd_for_esf(copy.deepcopy(scan_source_pcd))

    esf_target = np.asarray(
        compute_esf_descriptor(pcd_o3d=target_esf_pcd, exe_path=esf_exe_path),
        dtype=np.float32,
    )
    esf_source = np.asarray(
        compute_esf_descriptor(pcd_o3d=source_esf_pcd, exe_path=esf_exe_path),
        dtype=np.float32,
    )
    if esf_target.size != 640 or esf_source.size != 640:
        raise ValueError(
            f"Expected ESF length 640, got target={esf_target.size}, source={esf_source.size}"
        )

    norm_target = np.asarray(generate_normal_xray_hist(pcd=target_esf_pcd)[1], dtype=np.float32)
    norm_source = np.asarray(generate_normal_xray_hist(pcd=source_esf_pcd)[1], dtype=np.float32)
    if norm_target.size != 64 or norm_source.size != 64:
        raise ValueError(
            f"Expected normal-hist length 64, got target={norm_target.size}, source={norm_source.size}"
        )

    # Hauptvektor wie im Training
    de = np.abs(esf_target - esf_source).astype(np.float32)
    dn = np.abs(norm_target - norm_source).astype(np.float32)

    if legacy_normalize_main_diff:
        de = normalize_to_minus_one_and_one_v2(de).astype(np.float32)

    x704 = np.concatenate([de, dn]).astype(np.float32)

    # Extra 44D
    extra_feature_variant = str(extra_feature_variant).lower()

    if extra_feature_variant == "v1":
        esfA = esf_target.reshape(10, 64)
        esfB = esf_source.reshape(10, 64)
        extra44 = []
        for i in range(10):
            extra44.extend(_row_metrics_v1(esfA[i], esfB[i]))
        extra44.extend(_row_metrics_v1(norm_target, norm_source))
        x44 = np.asarray(extra44, dtype=np.float32)

    elif extra_feature_variant == "v2":
        x44 = _build_extra44_v2_precomputed_style(
            esf_target=esf_target,
            esf_source=esf_source,
            norm_target=norm_target,
            norm_source=norm_source,
        )

    else:
        raise ValueError(
            f"Unknown extra_feature_variant='{extra_feature_variant}'. Use 'v1' or 'v2'."
        )

    if x44.size != 44:
        raise ValueError(f"Expected extra feature length 44, got {x44.size}")

    # Grid 27D
    grid_vector = np.asarray(
        features_for_object(
            pcd_ref_o3d=model_target_pcd,
            pcd_scan_obj_o3d=scan_source_pcd,
        ),
        dtype=np.float32,
    )
    if grid_vector.size != 27:
        raise ValueError(f"Expected grid feature length 27, got {grid_vector.size}")

    xext = np.concatenate([x44, grid_vector]).astype(np.float32)

    return {
        "x704": x704,
        "x44": x44,
        "x27": grid_vector,
        "xext": xext,
        "extra_feature_variant": extra_feature_variant,
    }


def _normalize_state_dict_keys(state_dict: Dict[str, torch.Tensor], model_keys) -> Dict[str, torch.Tensor]:
    normalized = {}
    for k, v in state_dict.items():
        nk = k
        for pref in ("module.", "model.", "net."):
            if nk.startswith(pref):
                nk = nk[len(pref):]
        normalized[nk] = v

    model_key_set = set(model_keys)
    if any(k.startswith("m.") for k in model_key_set):
        normalized = {k if k.startswith("m.") else f"m.{k}": v for k, v in normalized.items()}
    return normalized

def load_inference_model(checkpoint_path: str, device: str = "cpu") -> torch.nn.Module:
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict):
        state_dict = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
    else:
        state_dict = ckpt

    out_dim = 2
    for k, v in state_dict.items():
        if k.endswith("head.4.weight") and hasattr(v, "shape") and len(v.shape) == 2:
            out_dim = int(v.shape[0])
            break

    base_model = MLPFlex(
        p=0.5,
        out_dim=out_dim,
        use_main=True,
        use_esf_norm=False,
        use_grid=True,
    )
    model = ChannelMaskWrapper(base_model, mode=FeatMode.MAIN_GRID, d_esf_norm=44, d_grid=27, d_main=704).to(device)

    norm_sd = _normalize_state_dict_keys(state_dict, model.state_dict().keys())
    missing, unexpected = model.load_state_dict(norm_sd, strict=False)
    print("[LOAD] missing:", missing)
    print("[LOAD] unexpected:", unexpected)

    if len(missing) > 0 or len(unexpected) > 0:
        raise RuntimeError(
            f"Checkpoint/Architektur mismatch | missing={len(missing)} unexpected={len(unexpected)}"
        )

    model.eval()
    return model

def load_inference_model_all_feats(checkpoint_path: str, device: str = "cpu") -> torch.nn.Module:
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict):
        state_dict = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
    else:
        state_dict = ckpt

    out_dim = 2
    for k, v in state_dict.items():
        if k.endswith("head.4.weight") and hasattr(v, "shape") and len(v.shape) == 2:
            out_dim = int(v.shape[0])
            break

    base_model = MLPFlex(
        p=0.5,
        out_dim=out_dim,
        use_main=True,
        use_esf_norm=True,
        use_grid=True,
    )
    model = ChannelMaskWrapper(base_model, mode=FeatMode.ALL, d_esf_norm=44, d_grid=27, d_main=704).to(device)

    norm_sd = _normalize_state_dict_keys(state_dict, model.state_dict().keys())
    missing, unexpected = model.load_state_dict(norm_sd, strict=False)
    print("[LOAD] missing:", missing)
    print("[LOAD] unexpected:", unexpected)

    if len(missing) > 0 or len(unexpected) > 0:
        raise RuntimeError(
            f"Checkpoint/Architektur mismatch | missing={len(missing)} unexpected={len(unexpected)}"
        )

    model.eval()
    return model


def run_inference(
    model: torch.nn.Module,
    x704: np.ndarray,
    xext: np.ndarray,
    t_star: float,
    device: str = "cpu"
) -> Dict[str, object]:

    with torch.no_grad():
        t704 = torch.from_numpy(x704).unsqueeze(0).to(device)
        text = torch.from_numpy(xext).unsqueeze(0).to(device)

        logits = model(t704, text)

        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        p1 = float(probs[1])

        # ENTSCHEIDUNG MIT THRESHOLD
        pred_idx = 1 if p1 >= t_star else 0

    return {
        "predicted_class": pred_idx,
        "probabilities": probs.tolist(),
        "confidence": p1,
        "threshold_used": t_star,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference for MLPFlex + ChannelMaskWrapper(FeatMode.ALL)")
    parser.add_argument("--model-target", help="Path to model target point cloud")
    parser.add_argument("--scan-source", help="Path to scan source point cloud")
    parser.add_argument("--checkpoint", help="Path to trained checkpoint")
    parser.add_argument("--esf-exe", help="Path to esf_estimation executable")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Inference device (default: auto)")
    parser.add_argument("--print-vectors", action="store_true", help="Print vector dimensions and first values")
    parser.add_argument(
        "--legacy-normalize-main-diff", default = True,
        action="store_true",
        help="Normalize only ESF abs-diff to [-1,1] (legacy behavior). Disabled by default for train parity.",
    )
    return parser.parse_args()


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def _resolve_device(device_arg: str) -> str:
    if device_arg == "cpu":
        return "cpu"
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but CUDA is not available")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_default_file(explicit_path: str, relative_dir: str, role: str, supported_exts) -> str:
    if explicit_path:
        return explicit_path

    search_dir = os.path.join(HERE, relative_dir)
    if not os.path.isdir(search_dir):
        raise FileNotFoundError(
            f"No --{role} provided and default directory does not exist: {search_dir}"
        )

    candidates = [
        entry
        for entry in os.listdir(search_dir)
        if os.path.isfile(os.path.join(search_dir, entry))
        and os.path.splitext(entry)[1].lower() in supported_exts
    ]

    if not candidates:
        raise FileNotFoundError(
            f"No --{role} provided and no supported files found in: {search_dir}"
        )

    selected = os.path.join(search_dir, candidates[0])
    logging.info("Auto-selected --%s: %s", role, selected)
    return selected


def main() -> None:
    _configure_logging()
    args = parse_args()
    device = _resolve_device(args.device)

    point_cloud_exts = {".ply", ".pcd", ".xyz", ".xyzn", ".xyzrgb", ".obj", ".stl", ".off", ".gltf", ".glb", ".las", ".txt", ".asc", ".csv"}
    checkpoint_exts = {".pth", ".pt", ".ckpt", ".bin"}
    esf_exe_exts = {".exe"}

    model_target_path = _resolve_default_file(args.model_target, os.path.join("input", "model"), "model-target", point_cloud_exts)
    scan_source_path = _resolve_default_file(args.scan_source, os.path.join("input", "scan"), "scan-source", point_cloud_exts)
    checkpoint_path = _resolve_default_file(args.checkpoint, "checkpoint", "checkpoint", checkpoint_exts)
    esf_exe_path = _resolve_default_file(args.esf_exe, ".", "esf-exe", esf_exe_exts)

    logging.info("Inference input | model-target=%s", model_target_path)
    logging.info("Inference input | scan-source=%s", scan_source_path)
    logging.info("Inference input | checkpoint=%s", checkpoint_path)
    logging.info("Inference input | esf-exe=%s", esf_exe_path)
    logging.info("Inference input | device-request=%s", args.device)
    logging.info("Inference input | device-selected=%s", device)
    logging.info("Inference input | print-vectors=%s", args.print_vectors)
    logging.info("Inference input | legacy-normalize-main-diff=%s", args.legacy_normalize_main_diff)

    target = read_pcd_in_any_format(model_target_path)
    source = read_pcd_in_any_format(scan_source_path)

    feats = build_input_vectors(
    target,
    source,
    esf_exe_path,
    legacy_normalize_main_diff=False,
    extra_feature_variant="v2",
)
    if args.print_vectors:
        print(f"x704: {feats['x704'].shape}, x44: {feats['x44'].shape}, x27: {feats['x27'].shape}, xext: {feats['xext'].shape}")
        print("x704[:8] =", np.round(feats["x704"][:8], 6).tolist())
        print("xext[:8] =", np.round(feats["xext"][:8], 6).tolist())

    model = load_inference_model(checkpoint_path, device=device)

    #A. Nur main
    #xext_zero = np.zeros_like(feats["xext"], dtype=np.float32)
    #pred = run_inference(model, feats["x704"], xext_zero, device=device, t_star=0.96)

    #B. Nur extra+grid, main auf 0
    #x704_zero = np.zeros_like(feats["x704"], dtype=np.float32)
    #pred = run_inference(model, x704_zero, feats["xext"], device=device, t_star=0.96)

    #C. Nur extra ohne grid
    #xext_extra_only = np.concatenate([feats["xext"][:44], np.zeros_like(feats["xext"][44:72])], axis=0).astype(np.float32)
    #pred = run_inference(model, feats["x704"], xext_extra_only, device=device, t_star=0.96)

    #D. Nur grid ohne extra
    xext_grid_only = np.concatenate([np.zeros_like(feats["xext"][:44]), feats["xext"][44:72]], axis=0).astype(np.float32)
    pred = run_inference(model, feats["x704"], xext_grid_only, device=device, t_star=0.96)

    #E. Nur grid alleine
    #x704_zero = np.zeros_like(feats["x704"], dtype=np.float32)
    #xext_grid_only = np.concatenate([np.zeros_like(feats["xext"][:44]), feats["xext"][44:72]], axis=0).astype(np.float32)
    #pred = run_inference(model, x704_zero , xext_grid_only, device=device, t_star=0.96)

    # All Features
    #pred = run_inference(model, feats["x704"], feats["xext"], device=device, t_star=0.96)
    print(pred)
    print(json.dumps(pred, indent=2))
    if pred["predicted_class"] == 0:
        if np.array(source.points).shape[0] < 1000:
            print("Object Verification: Scan has to less points")
        else:
            print("Object Verification: Scan is another Object")
    else:
        print("Object Verification: Scan is Reference Object")

if __name__ == "__main__":
    main()
