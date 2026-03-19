import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import open3d as o3d


def _load_transform(path: Path) -> np.ndarray:
    text = path.read_text(encoding="utf-8").strip()
    try:
        obj = json.loads(text)
        arr = np.asarray(obj, dtype=np.float64)
    except json.JSONDecodeError:
        vals = [float(v) for v in text.replace(",", " ").split()]
        arr = np.asarray(vals, dtype=np.float64)
    if arr.size != 16:
        raise ValueError(f"Transform must contain 16 values, got {arr.size}: {path}")
    return arr.reshape(4, 4)


def _rot_y_180() -> np.ndarray:
    r = np.eye(4)
    r[0, 0] = -1.0
    r[2, 2] = -1.0
    return r


def _rot_z_180() -> np.ndarray:
    r = np.eye(4)
    r[0, 0] = -1.0
    r[1, 1] = -1.0
    return r


def _candidate_transforms(ppf_transform: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    return [
        ("ppf_raw", ppf_transform),
        ("ppf_raw_plus_ry180", ppf_transform @ _rot_y_180()),
        ("ppf_raw_plus_rz180", ppf_transform @ _rot_z_180()),
    ]


def _score_transform(source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud, transform: np.ndarray) -> float:
    src = source.voxel_down_sample(0.03)
    tgt = target.voxel_down_sample(0.03)
    src = src.transform(transform.copy())
    d = np.asarray(src.compute_point_cloud_distance(tgt), dtype=np.float64)
    if d.size == 0:
        return float("inf")
    return float(np.mean(np.clip(d, 0.0, 0.2)))


def run(source_file: Path, target_file: Path, ppf_transform: np.ndarray, out_dir: Path, icp_distance_threshold: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    source = o3d.io.read_point_cloud(str(source_file))
    target = o3d.io.read_point_cloud(str(target_file))

    if len(source.points) == 0 or len(target.points) == 0:
        raise ValueError("Source/Target point cloud is empty.")

    candidates = _candidate_transforms(ppf_transform)
    scored = []
    for name, t in candidates:
        scored.append((name, t, _score_transform(source, target, t)))

    best_name, best_transform, best_score = sorted(scored, key=lambda x: x[2])[0]

    source_ppf = o3d.geometry.PointCloud(source)
    source_ppf.transform(best_transform)
    ppf_out = out_dir / "source_after_ppf_before_icp.ply"
    o3d.io.write_point_cloud(str(ppf_out), source_ppf)

    ppf_meta = {
        "best_candidate": best_name,
        "score_mean_clipped_distance": best_score,
        "candidate_scores": [{"name": n, "score": s} for n, _, s in scored],
        "transform_best": best_transform.tolist(),
    }
    (out_dir / "ppf_selection.json").write_text(json.dumps(ppf_meta, indent=2), encoding="utf-8")

    reg = o3d.pipelines.registration.registration_icp(
        source,
        target,
        icp_distance_threshold,
        best_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )

    source_icp = o3d.geometry.PointCloud(source)
    source_icp.transform(reg.transformation)
    o3d.io.write_point_cloud(str(out_dir / "source_after_icp.ply"), source_icp)

    icp_meta = {
        "fitness": float(reg.fitness),
        "inlier_rmse": float(reg.inlier_rmse),
        "transform_icp": reg.transformation.tolist(),
    }
    (out_dir / "icp_result.json").write_text(json.dumps(icp_meta, indent=2), encoding="utf-8")

    print(f"Saved pre-ICP PLY: {ppf_out}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Save PPF-aligned PLY before ICP and run ICP refinement.")
    p.add_argument("--source", required=True, help="As-built/source point cloud")
    p.add_argument("--target", required=True, help="As-planned/target point cloud")
    p.add_argument("--ppf-transform", required=True, help="Path to 4x4 PPF transform (json or text with 16 values)")
    p.add_argument("--out-dir", default="artefact_02_validation/registration_outputs")
    p.add_argument("--icp-threshold", type=float, default=0.08)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        source_file=Path(args.source),
        target_file=Path(args.target),
        ppf_transform=_load_transform(Path(args.ppf_transform)),
        out_dir=Path(args.out_dir),
        icp_distance_threshold=args.icp_threshold,
    )
