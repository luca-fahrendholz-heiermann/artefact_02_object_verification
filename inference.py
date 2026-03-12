import argparse
import copy
import json
import os
import sys
from typing import Dict, Tuple, Optional, List, Any
from torch.utils.data import Dataset, DataLoader
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


def build_input_vectors(model_target_pcd: o3d.geometry.PointCloud,
                        scan_source_pcd: o3d.geometry.PointCloud,
                        esf_exe_path: str,
                        legacy_normalize_main_diff: bool = False) -> Dict[str, np.ndarray]:
    target_esf_pcd = preprocess_pcd_for_esf(copy.deepcopy(model_target_pcd))
    source_esf_pcd = preprocess_pcd_for_esf(copy.deepcopy(scan_source_pcd))

    esf_target = np.asarray(compute_esf_descriptor(pcd_o3d=target_esf_pcd, exe_path=esf_exe_path), dtype=np.float32)
    esf_source = np.asarray(compute_esf_descriptor(pcd_o3d=source_esf_pcd, exe_path=esf_exe_path), dtype=np.float32)
    if esf_target.size != 640 or esf_source.size != 640:
        raise ValueError(f"Expected ESF length 640, got target={esf_target.size}, source={esf_source.size}")

    norm_target = np.asarray(generate_normal_xray_hist(pcd=target_esf_pcd)[1], dtype=np.float32)
    norm_source = np.asarray(generate_normal_xray_hist(pcd=source_esf_pcd)[1], dtype=np.float32)
    if norm_target.size != 64 or norm_source.size != 64:
        raise ValueError(f"Expected normal-hist length 64, got target={norm_target.size}, source={norm_source.size}")

    # Training-Parität (ESFRefPairDatasetMLP704Fast):
    #   de = abs(esf_ref - esf_scan), dn = abs(norm_ref - norm_scan)
    #   keine weitere Normalisierung auf dem 704er Hauptvektor.
    de = np.abs(esf_target - esf_source).astype(np.float32)
    dn = np.abs(norm_target - norm_source).astype(np.float32)

    # Optionaler Legacy-Pfad für ältere Pipelines, die nur ESF-Diff auf [-1,1] skaliert haben.
    # Standard bleibt FALSE, damit das Verhalten dem Training entspricht.
    if legacy_normalize_main_diff:
        de = normalize_to_minus_one_and_one_v2(de).astype(np.float32)

    x704 = np.concatenate([de, dn]).astype(np.float32)

    esfA = esf_target.reshape(10, 64)
    esfB = esf_source.reshape(10, 64)
    extra44 = []
    for i in range(10):
        extra44.extend(_row_metrics(esfA[i], esfB[i]))
    extra44.extend(_row_metrics(norm_target, norm_source))
    x44 = np.asarray(extra44, dtype=np.float32)
    if x44.size != 44:
        raise ValueError(f"Expected extra feature length 44, got {x44.size}")

    grid_vector = np.asarray(
        features_for_object(pcd_ref_o3d=model_target_pcd, pcd_scan_obj_o3d=scan_source_pcd),
        dtype=np.float32,
    )
    if grid_vector.size != 27:
        raise ValueError(f"Expected grid feature length 27, got {grid_vector.size}")

    xext = np.concatenate([x44, grid_vector]).astype(np.float32)
    return {"x704": x704, "x44": x44, "x27": grid_vector, "xext": xext}


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
        use_esf_norm=True,
        use_grid=True,
    )
    model = ChannelMaskWrapper(base_model, mode=FeatMode.ALL, d_esf_norm=44, d_grid=27, d_main=704).to(device)

    norm_sd = _normalize_state_dict_keys(state_dict, model.state_dict().keys())
    missing, unexpected = model.load_state_dict(norm_sd, strict=False)
    if missing:
        print(f"[WARN] Missing keys during load: {len(missing)}")
    if unexpected:
        print(f"[WARN] Unexpected keys during load: {len(unexpected)}")

    model.eval()
    return model


def run_inference(model: torch.nn.Module, x704: np.ndarray, xext: np.ndarray, device: str = "cpu") -> Dict[str, object]:
    with torch.no_grad():
        t704 = torch.from_numpy(x704).unsqueeze(0).to(device)
        text = torch.from_numpy(xext).unsqueeze(0).to(device)
        logits = model(t704, text)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
    return {
        "predicted_class": pred_idx,
        "probabilities": probs.tolist(),
        "confidence": float(probs[pred_idx]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference for MLPFlex + ChannelMaskWrapper(FeatMode.ALL)")
    parser.add_argument("--model-target", required=True, help="Path to model target point cloud")
    parser.add_argument("--scan-source", required=True, help="Path to scan source point cloud")
    parser.add_argument("--checkpoint", default=os.path.join(HERE, "checkpoint", "obj_verf_3classes_tanh_ep_300_lr_0_001_6_layer_fold_4_best_model.pth"), help="Path to trained checkpoint")
    parser.add_argument("--esf-exe", default=os.path.join(HERE, "esf_estimation.exe"), help="Path to esf_estimation executable")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Inference device")
    parser.add_argument("--print-vectors", action="store_true", help="Print vector dimensions and first values")
    parser.add_argument(
        "--legacy-normalize-main-diff",
        action="store_true",
        help="Normalize only ESF abs-diff to [-1,1] (legacy behavior). Disabled by default for train parity.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"

    target = read_pcd_in_any_format(args.model_target)
    source = read_pcd_in_any_format(args.scan_source)

    feats = build_input_vectors(
        target,
        source,
        args.esf_exe,
        legacy_normalize_main_diff=args.legacy_normalize_main_diff,
    )
    if args.print_vectors:
        print(f"x704: {feats['x704'].shape}, x44: {feats['x44'].shape}, x27: {feats['x27'].shape}, xext: {feats['xext'].shape}")
        print("x704[:8] =", np.round(feats["x704"][:8], 6).tolist())
        print("xext[:8] =", np.round(feats["xext"][:8], 6).tolist())

    model = load_inference_model(args.checkpoint, device=device)
    pred = run_inference(model, feats["x704"], feats["xext"], device=device)

    print(json.dumps(pred, indent=2))


if __name__ == "__main__":
    main()
