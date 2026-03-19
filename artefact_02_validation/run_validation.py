import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import open3d as o3d
import pandas as pd
from sklearn.neighbors import KDTree

from ai_model_inference.inference import (
    build_input_vectors,
    load_inference_model,
    read_pcd_in_any_format,
    run_inference,
)

SUPPORTED_PCD_EXTS = {".ply", ".pcd", ".xyz", ".xyzn", ".xyzrgb", ".obj", ".stl", ".off", ".gltf", ".glb", ".las", ".txt", ".asc", ".csv"}
DOMAIN_TO_FOLDER = {
    "indoor": "Indoor_Production_QC",
    "outdoor": "Outdoor_Construction_Site_PM",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_checkpoint_metadata(checkpoint: Path) -> Dict[str, object]:
    candidates = list(checkpoint.parent.glob("*best_model_info.json"))
    metadata: Dict[str, object] = {}
    if candidates:
        with candidates[0].open("r", encoding="utf-8") as f:
            metadata = json.load(f)
    metadata["checkpoint_path"] = str(checkpoint)
    metadata["checkpoint_sha256"] = _sha256(checkpoint)
    return metadata


def _list_point_clouds(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    files = [
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_PCD_EXTS and p.name.lower() != "placeholder.txt"
    ]
    return sorted(files)


def _extract_points_by_knn(
    assembly_part_path: Path,
    source_scan_path: Path,
    extracted_dir: Path,
    knn_neighbors: int,
) -> Tuple[Path, int]:
    pcd_part = read_pcd_in_any_format(str(assembly_part_path))
    pcd_source = read_pcd_in_any_format(str(source_scan_path))

    part_pts = np.asarray(pcd_part.points)
    source_pts = np.asarray(pcd_source.points)

    if source_pts.shape[0] == 0:
        out_name = f"{assembly_part_path.stem}_extracted.ply"
        out_path = extracted_dir / out_name
        o3d.io.write_point_cloud(str(out_path), o3d.geometry.PointCloud())
        return out_path, 0

    k = max(1, min(knn_neighbors, source_pts.shape[0]))
    tree = KDTree(source_pts)
    _, indices = tree.query(part_pts, k=k, return_distance=True, sort_results=True)
    keep_idx = np.unique(indices.reshape(-1))

    pcd_extracted = pcd_source.select_by_index(keep_idx.tolist())
    out_name = f"{assembly_part_path.stem}_extracted.ply"
    out_path = extracted_dir / out_name
    extracted_dir.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(out_path), pcd_extracted)
    return out_path, int(len(keep_idx))


def _run_single_comparison(
    model,
    target_path: Path,
    source_path: Path,
    esf_exe: Path,
    threshold: float,
    device: str,
) -> Dict[str, object]:
    target_pcd = read_pcd_in_any_format(str(target_path))
    source_pcd = read_pcd_in_any_format(str(source_path))

    feats = build_input_vectors(
        model_target_pcd=target_pcd,
        scan_source_pcd=source_pcd,
        esf_exe_path=str(esf_exe),
        legacy_normalize_main_diff=False,
        extra_feature_variant="v2",
    )
    pred = run_inference(
        model=model,
        x704=feats["x704"],
        xext=feats["xext"],
        t_star=threshold,
        device=device,
    )
    return pred


def run_validation(
    domain: str,
    root_dir: Path,
    checkpoint: Path,
    esf_exe: Path,
    threshold: float,
    knn_neighbors: int,
    device: str,
) -> Path:
    domain_folder = DOMAIN_TO_FOLDER[domain]
    base_dir = root_dir / domain_folder
    if not base_dir.exists():
        raise FileNotFoundError(f"Domain folder not found: {base_dir}")

    model = load_inference_model(str(checkpoint), device=device)
    ckpt_meta = _load_checkpoint_metadata(checkpoint)

    all_rows: List[Dict[str, object]] = []

    project_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir()])
    for project_dir in project_dirs:
        target_files = _list_point_clouds(project_dir / "target")
        source_files = _list_point_clouds(project_dir / "source")
        assembly_parts = _list_point_clouds(project_dir / "assembly_parts")

        if not target_files or not source_files:
            continue

        target_path = target_files[0]

        for source_path in source_files:
            global_pred = _run_single_comparison(
                model=model,
                target_path=target_path,
                source_path=source_path,
                esf_exe=esf_exe,
                threshold=threshold,
                device=device,
            )
            all_rows.append(
                {
                    "domain": domain_folder,
                    "project": project_dir.name,
                    "comparison_scope": "global_target_vs_source",
                    "source_file": str(source_path),
                    "reference_file": str(target_path),
                    "extracted_file": "",
                    "extracted_points": None,
                    "predicted_class": int(global_pred["predicted_class"]),
                    "probability_class_0": float(global_pred["probabilities"][0]),
                    "probability_class_1": float(global_pred["probabilities"][1]),
                    "confidence": float(global_pred["confidence"]),
                    "threshold_used": float(global_pred["threshold_used"]),
                    "ground_truth": None,
                    "checkpoint_path": ckpt_meta["checkpoint_path"],
                    "checkpoint_sha256": ckpt_meta["checkpoint_sha256"],
                    "model_metadata_json": json.dumps(ckpt_meta, ensure_ascii=False),
                }
            )

            extracted_dir = project_dir / "extracted"
            for assembly_path in assembly_parts:
                extracted_path, n_pts = _extract_points_by_knn(
                    assembly_part_path=assembly_path,
                    source_scan_path=source_path,
                    extracted_dir=extracted_dir,
                    knn_neighbors=knn_neighbors,
                )

                part_pred = _run_single_comparison(
                    model=model,
                    target_path=assembly_path,
                    source_path=extracted_path,
                    esf_exe=esf_exe,
                    threshold=threshold,
                    device=device,
                )
                all_rows.append(
                    {
                        "domain": domain_folder,
                        "project": project_dir.name,
                        "comparison_scope": "assembly_part_vs_extracted",
                        "source_file": str(source_path),
                        "reference_file": str(assembly_path),
                        "extracted_file": str(extracted_path),
                        "extracted_points": n_pts,
                        "predicted_class": int(part_pred["predicted_class"]),
                        "probability_class_0": float(part_pred["probabilities"][0]),
                        "probability_class_1": float(part_pred["probabilities"][1]),
                        "confidence": float(part_pred["confidence"]),
                        "threshold_used": float(part_pred["threshold_used"]),
                        "ground_truth": None,
                        "checkpoint_path": ckpt_meta["checkpoint_path"],
                        "checkpoint_sha256": ckpt_meta["checkpoint_sha256"],
                        "model_metadata_json": json.dumps(ckpt_meta, ensure_ascii=False),
                    }
                )

    if not all_rows:
        raise RuntimeError(f"No validation pairs found in {base_dir}")

    df = pd.DataFrame(all_rows)

    out_root = root_dir / "results" / domain_folder
    out_root.mkdir(parents=True, exist_ok=True)

    all_file = out_root / f"{domain_folder}_all_projects_predictions.xlsx"
    df.to_excel(all_file, index=False)

    all_json = out_root / f"{domain_folder}_all_projects_predictions.json"
    with all_json.open("w", encoding="utf-8") as f:
        json.dump(all_rows, f, indent=2, ensure_ascii=False)

    for project, df_project in df.groupby("project"):
        project_dir = out_root / project
        project_dir.mkdir(parents=True, exist_ok=True)
        df_project.to_excel(project_dir / f"{project}_predictions.xlsx", index=False)
        df_project.to_json(project_dir / f"{project}_predictions.json", orient="records", indent=2, force_ascii=False)

    print(f"Saved validation predictions to: {all_file}")
    return all_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch validation for indoor/outdoor projects.")
    parser.add_argument("--domain", required=True, choices=["indoor", "outdoor"])
    parser.add_argument("--root-dir", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--checkpoint", default=str(Path("ai_model_inference/checkpoint/obj_verf_main_grid_big_best_model_f5.pth")))
    parser.add_argument("--esf-exe", default=str(Path("ai_model_inference/esf_estimation.exe")))
    parser.add_argument("--threshold", type=float, default=0.96)
    parser.add_argument("--knn-neighbors", type=int, default=10)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_validation(
        domain=args.domain,
        root_dir=Path(args.root_dir),
        checkpoint=Path(args.checkpoint),
        esf_exe=Path(args.esf_exe),
        threshold=args.threshold,
        knn_neighbors=args.knn_neighbors,
        device=args.device,
    )
