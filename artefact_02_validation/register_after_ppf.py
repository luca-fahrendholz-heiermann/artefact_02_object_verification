import argparse
import json
import math
from pathlib import Path
from typing import List, Tuple

import numpy as np
import open3d as o3d
import pandas as pd


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


def _candidate_transforms(ppf_transform: np.ndarray, include_inverse: bool = True) -> List[Tuple[str, np.ndarray]]:
    candidates = [
        ("ppf_raw", ppf_transform),
        ("ppf_raw_plus_ry180", ppf_transform @ _rot_y_180()),
        ("ppf_raw_plus_rz180", ppf_transform @ _rot_z_180()),
    ]
    if include_inverse:
        inv = np.linalg.inv(ppf_transform)
        candidates.extend(
            [
                ("ppf_inv", inv),
                ("ppf_inv_plus_ry180", inv @ _rot_y_180()),
                ("ppf_inv_plus_rz180", inv @ _rot_z_180()),
            ]
        )
    return candidates


def _score_transform(source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud, transform: np.ndarray) -> float:
    src = source.voxel_down_sample(0.03)
    tgt = target.voxel_down_sample(0.03)
    src = src.transform(transform.copy())
    d = np.asarray(src.compute_point_cloud_distance(tgt), dtype=np.float64)
    if d.size == 0:
        return float("inf")
    return float(np.mean(np.clip(d, 0.0, 0.2)))


def _orthonormalize_transform(T: np.ndarray) -> np.ndarray:
    Tout = np.array(T, dtype=np.float64, copy=True)
    R = Tout[:3, :3]
    U, _, Vt = np.linalg.svd(R)
    R_ortho = U @ Vt
    if np.linalg.det(R_ortho) < 0:
        U[:, -1] *= -1
        R_ortho = U @ Vt
    Tout[:3, :3] = R_ortho
    return Tout


def _to_centered_frame(T_world: np.ndarray, source_center: np.ndarray, target_center: np.ndarray) -> np.ndarray:
    C_s = np.eye(4, dtype=np.float64)
    C_t = np.eye(4, dtype=np.float64)
    C_s[:3, 3] = source_center
    C_t[:3, 3] = target_center
    return np.linalg.inv(C_t) @ T_world @ C_s


def _write_matrix_txt(path: Path, T: np.ndarray) -> None:
    lines = []
    for r in range(4):
        lines.append(" ".join(f"{float(v):.10f}" for v in T[r, :]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _ensure_normals(pcd: o3d.geometry.PointCloud, radius: float = 0.2, max_nn: int = 30) -> None:
    if pcd.has_normals():
        return
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    pcd.normalize_normals()


def _matrix_to_xyz_rot_deg(T: np.ndarray) -> Tuple[float, float, float]:
    R = np.asarray(T[:3, :3], dtype=np.float64)
    sy = -R[2, 0]
    sy = np.clip(sy, -1.0, 1.0)
    y = math.asin(sy)
    cy = math.cos(y)
    if abs(cy) < 1e-8:
        x = math.atan2(-R[1, 2], R[1, 1])
        z = 0.0
    else:
        x = math.atan2(R[2, 1], R[2, 2])
        z = math.atan2(R[1, 0], R[0, 0])
    return math.degrees(x), math.degrees(y), math.degrees(z)


def _select_landmark_indices(points: np.ndarray) -> dict:
    idx = {
        "minX": int(np.argmin(points[:, 0])),
        "maxX": int(np.argmax(points[:, 0])),
        "minY": int(np.argmin(points[:, 1])),
        "maxY": int(np.argmax(points[:, 1])),
        "minZ": int(np.argmin(points[:, 2])),
        "maxZ": int(np.argmax(points[:, 2])),
    }
    return idx


def _angle_diff_deg(a: float, b: float) -> float:
    d = b - a
    while d > math.pi:
        d -= 2 * math.pi
    while d < -math.pi:
        d += 2 * math.pi
    return math.degrees(d)


def _landmark_rotation_from_centers(points_before: np.ndarray, points_after: np.ndarray, landmark_idx: dict) -> Tuple[dict, pd.DataFrame]:
    c0 = points_before.mean(axis=0)
    c1 = points_after.mean(axis=0)
    rows = []
    x_angles = []
    y_angles = []
    z_angles = []
    for name, i in landmark_idx.items():
        v0 = points_before[i] - c0
        v1 = points_after[i] - c1

        a0_x = math.atan2(v0[2], v0[1])
        a1_x = math.atan2(v1[2], v1[1])
        dx = _angle_diff_deg(a0_x, a1_x)

        a0_y = math.atan2(v0[2], v0[0])
        a1_y = math.atan2(v1[2], v1[0])
        dy = _angle_diff_deg(a0_y, a1_y)

        a0_z = math.atan2(v0[1], v0[0])
        a1_z = math.atan2(v1[1], v1[0])
        dz = _angle_diff_deg(a0_z, a1_z)

        x_angles.append(dx)
        y_angles.append(dy)
        z_angles.append(dz)
        rows.append(
            {
                "landmark": name,
                "index": int(i),
                "dx_deg_yz": dx,
                "dy_deg_xz": dy,
                "dz_deg_xy": dz,
                "vx_before": float(v0[0]),
                "vy_before": float(v0[1]),
                "vz_before": float(v0[2]),
                "vx_after": float(v1[0]),
                "vy_after": float(v1[1]),
                "vz_after": float(v1[2]),
            }
        )

    detail_df = pd.DataFrame(rows)
    robust = {
        "x_rot": float(np.median(x_angles)) if x_angles else 0.0,
        "y_rot": float(np.median(y_angles)) if y_angles else 0.0,
        "z_rot": float(np.median(z_angles)) if z_angles else 0.0,
    }
    return robust, detail_df


def _transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    ones = np.ones((points.shape[0], 1), dtype=np.float64)
    p = np.concatenate([points.astype(np.float64), ones], axis=1)
    out = (T @ p.T).T
    return out[:, :3]


def _build_landmark_excel(
    source_points: np.ndarray,
    T_noise: np.ndarray,
    T_est: np.ndarray,
    out_xlsx: Path,
) -> None:
    landmark_idx = _select_landmark_indices(source_points)
    center_before = source_points.mean(axis=0)
    transformed_est = _transform_points(source_points, T_est)
    center_after = transformed_est.mean(axis=0)
    trans_eff = center_after - center_before
    rot_eff, detail_df = _landmark_rotation_from_centers(source_points, transformed_est, landmark_idx)

    T_noise_inv = np.linalg.inv(T_noise)
    T_diff = T_est @ T_noise

    rows = []
    for name, T in [
        ("T_noise", T_noise),
        ("inv(T_noise)_ideal", T_noise_inv),
        ("T_est", T_est),
        ("T_diff=T_est@T_noise", T_diff),
    ]:
        rx, ry, rz = _matrix_to_xyz_rot_deg(T)
        rows.append(
            {
                "Methode": name,
                "x_trans": float(T[0, 3]),
                "y_trans": float(T[1, 3]),
                "z_trans": float(T[2, 3]),
                "x_rot": rx,
                "y_rot": ry,
                "z_rot": rz,
            }
        )

    rows.append(
        {
            "Methode": "Landmarken_Ansatz",
            "x_trans": float(trans_eff[0]),
            "y_trans": float(trans_eff[1]),
            "z_trans": float(trans_eff[2]),
            "x_rot": rot_eff["x_rot"],
            "y_rot": rot_eff["y_rot"],
            "z_rot": rot_eff["z_rot"],
        }
    )
    summary_df = pd.DataFrame(rows)

    matrix_rows = []
    for name, T in [
        ("T_noise", T_noise),
        ("inv(T_noise)_ideal", T_noise_inv),
        ("T_est", T_est),
        ("T_diff=T_est@T_noise", T_diff),
    ]:
        for r in range(4):
            matrix_rows.append(
                {
                    "Methode": name,
                    "row": r,
                    "c0": float(T[r, 0]),
                    "c1": float(T[r, 1]),
                    "c2": float(T[r, 2]),
                    "c3": float(T[r, 3]),
                }
            )
    matrices_df = pd.DataFrame(matrix_rows)

    ansatz_df = pd.DataFrame(
        [
            {"Schritt": "Landmarken fix im Initialzustand", "Wert": str(landmark_idx)},
            {"Schritt": "Center_before", "Wert": center_before.tolist()},
            {"Schritt": "Center_after(T_est)", "Wert": center_after.tolist()},
            {"Schritt": "Translation_effektiv", "Wert": trans_eff.tolist()},
            {"Schritt": "Rotation_effektiv_deg(x,y,z)", "Wert": [rot_eff["x_rot"], rot_eff["y_rot"], rot_eff["z_rot"]]},
        ]
    )

    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Uebersicht", index=False)
        matrices_df.to_excel(writer, sheet_name="Matrizen", index=False)
        detail_df.to_excel(writer, sheet_name="Landmarken_Details", index=False)
        ansatz_df.to_excel(writer, sheet_name="Ansatz", index=False)


def run(
    source_file: Path,
    target_file: Path,
    ppf_transform: np.ndarray,
    out_dir: Path,
    icp_distance_threshold: float,
    include_inverse: bool,
    save_all_candidates: bool,
    t_noise: np.ndarray = None,
    excel_out: Path = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    source = o3d.io.read_point_cloud(str(source_file))
    target = o3d.io.read_point_cloud(str(target_file))

    if len(source.points) == 0 or len(target.points) == 0:
        raise ValueError("Source/Target point cloud is empty.")

    candidates = _candidate_transforms(ppf_transform, include_inverse=include_inverse)
    scored = []
    for name, t in candidates:
        score = _score_transform(source, target, t)
        scored.append((name, t, score))
        if save_all_candidates:
            p = o3d.geometry.PointCloud(source)
            p.transform(t)
            o3d.io.write_point_cloud(str(out_dir / f"{name}_before_icp.ply"), p)

    best_name, best_transform_raw, best_score = sorted(scored, key=lambda x: x[2])[0]
    best_transform = _orthonormalize_transform(best_transform_raw)
    source_center = np.mean(np.asarray(source.points), axis=0).astype(np.float64)
    target_center = np.mean(np.asarray(target.points), axis=0).astype(np.float64)
    best_transform_centered = _to_centered_frame(best_transform, source_center, target_center)

    source_ppf = o3d.geometry.PointCloud(source)
    source_ppf.transform(best_transform)
    ppf_out = out_dir / "source_after_ppf_before_icp.ply"
    ok = o3d.io.write_point_cloud(str(ppf_out), source_ppf)
    if not ok:
        raise IOError(f"Could not write pre-ICP file: {ppf_out}")

    ppf_meta = {
        "best_candidate": best_name,
        "score_mean_clipped_distance": best_score,
        "candidate_scores": [{"name": n, "score": s} for n, _, s in scored],
        "transform_best_raw": best_transform_raw.tolist(),
        "transform_best": best_transform.tolist(),
        "transform_best_centered": best_transform_centered.tolist(),
        "source_center": source_center.tolist(),
        "target_center": target_center.tolist(),
    }
    (out_dir / "ppf_selection.json").write_text(json.dumps(ppf_meta, indent=2), encoding="utf-8")
    _write_matrix_txt(out_dir / "T_est_world.txt", best_transform)
    _write_matrix_txt(out_dir / "T_est_centered.txt", best_transform_centered)

    try:
        src_icp = o3d.geometry.PointCloud(source)
        tgt_icp = o3d.geometry.PointCloud(target)
        _ensure_normals(src_icp)
        _ensure_normals(tgt_icp)

        reg_p2l = o3d.pipelines.registration.registration_icp(
            src_icp,
            tgt_icp,
            icp_distance_threshold,
            best_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        )
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source,
            target,
            icp_distance_threshold,
            best_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        )

        reg = reg_p2l
        method = "point_to_plane"
        if (reg_p2p.fitness > reg_p2l.fitness) or (
            abs(reg_p2p.fitness - reg_p2l.fitness) < 1e-9 and reg_p2p.inlier_rmse < reg_p2l.inlier_rmse
        ):
            reg = reg_p2p
            method = "point_to_point"

        source_icp = o3d.geometry.PointCloud(source)
        source_icp.transform(reg.transformation)
        o3d.io.write_point_cloud(str(out_dir / "source_after_icp.ply"), source_icp)

        icp_meta = {
            "method_selected": method,
            "fitness_point_to_plane": float(reg_p2l.fitness),
            "inlier_rmse_point_to_plane": float(reg_p2l.inlier_rmse),
            "fitness_point_to_point": float(reg_p2p.fitness),
            "inlier_rmse_point_to_point": float(reg_p2p.inlier_rmse),
            "fitness": float(reg.fitness),
            "inlier_rmse": float(reg.inlier_rmse),
            "transform_icp": reg.transformation.tolist(),
        }
        (out_dir / "icp_result.json").write_text(json.dumps(icp_meta, indent=2), encoding="utf-8")
    except Exception as exc:
        (out_dir / "icp_result.json").write_text(
            json.dumps({"error": str(exc), "note": "Pre-ICP file was still written."}, indent=2),
            encoding="utf-8",
        )

    print(f"Saved pre-ICP PLY: {ppf_out.resolve()}")
    if t_noise is not None:
        excel_file = excel_out if excel_out is not None else (out_dir / "landmark_validation.xlsx")
        _build_landmark_excel(
            source_points=np.asarray(source.points, dtype=np.float64),
            T_noise=t_noise,
            T_est=best_transform,
            out_xlsx=excel_file,
        )
        print(f"Saved landmark validation workbook: {excel_file.resolve()}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Save PPF-aligned PLY before ICP and run ICP refinement.")
    p.add_argument("--source", required=True, help="As-built/source point cloud")
    p.add_argument("--target", required=True, help="As-planned/target point cloud")
    p.add_argument("--ppf-transform", required=True, help="Path to 4x4 PPF transform (json or text with 16 values)")
    p.add_argument("--out-dir", default="artefact_02_validation/registration_outputs")
    p.add_argument("--icp-threshold", type=float, default=0.08)
    p.add_argument("--no-inverse-check", action="store_true", help="Disable inverse-transform symmetry candidates.")
    p.add_argument("--save-all-candidates", action="store_true", help="Save all tested PPF candidate alignments as PLY files.")
    p.add_argument("--t-noise", default=None, help="Optional 4x4 noise transform for extended matrix+landmark validation.")
    p.add_argument("--excel-out", default=None, help="Optional output xlsx path for extended validation workbook.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        source_file=Path(args.source),
        target_file=Path(args.target),
        ppf_transform=_load_transform(Path(args.ppf_transform)),
        out_dir=Path(args.out_dir),
        icp_distance_threshold=args.icp_threshold,
        include_inverse=not args.no_inverse_check,
        save_all_candidates=args.save_all_candidates,
        t_noise=_load_transform(Path(args.t_noise)) if args.t_noise else None,
        excel_out=Path(args.excel_out) if args.excel_out else None,
    )
