#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Grid-Features (27D) als JSONL-Shards, wiederaufnahmefähig und atomar.

• Label==1: Features aus vorab erzeugtem Grid-JSON übernehmen.
• Label!=1: Features neu berechnen (Ref=100 %, Scan anhand perc & idx croppen).
• Shards: features_stream/<fold>-<split>.partNNN.jsonl (+ .tmp während Schreibens).
• State: features_stream/<fold>-<split>.state.json (Resume).
"""

import os, json, argparse, traceback, orjson
from typing import Dict, Any, List, Tuple
import numpy as np
from tqdm import tqdm
import open3d as o3d
import os
import json
import time
import copy
from sklearn.neighbors import KDTree
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, Optional, List

# ---- Importiere deine Grid-Funktionsbausteine (aus deinem Generator) ----
# (Die Datei 'generate_grid_comparison_json_augmentations.py' liegt laut Angabe im Projektroot.)

def uniform_subsample_point_cloud(pcd = None, points =None, num_points_to_select = 200000):
    """
    Uniformly subsample a point cloud.

    Parameters:
    - pcd: NumPy array representing the original point cloud with shape (N, 3).
    - num_points_to_select: Desired number of points in the subsampled point cloud.

    Returns:
    - subsampled_pcd: Subsampled point cloud as a NumPy array with shape (num_points_to_select, 3).
    """
    # Calculate the total number of points in the original point cloud

    if pcd == None:
        num_points_original = points.shape[0]
    else:
        points = np.array(pcd.points)
        num_points_original = points.shape[0]

    #print("SHAPE BEFORE: ",num_points_original)

    # Generate random indices to select points uniformly
    # if pcd.shape[0] < num_points_to_select:
    #     #print("FILL")
    #     pcd = fill_points(pcd , num_points_to_select)
    #     num_points_original = pcd.shape[0]
    #     #print("SHAPE AFER FILL: ", pcd.shape[0])

    if num_points_original > num_points_to_select:
        random_indices = np.random.choice(num_points_original, num_points_to_select, replace=False)


    # Select the corresponding points using the random indices
    if pcd == None:
        subsampled_pcd = points[random_indices]
    #print(len(subsampled_pcd))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(subsampled_pcd)
    else:
        pcd = pcd.select_by_index(random_indices)
    return pcd

def jitter_pcd(pcd, sigma=0.01, clip=0.05):
    pts = np.array(pcd.points)
    pts += np.clip(sigma * np.random.randn(*pts.shape), -1 * clip, clip)
    pcd_ = o3d.geometry.PointCloud()
    pcd_.points = o3d.utility.Vector3dVector(pts)
    return pcd_
def crop_group_point_cloud_by_percentage_z(pcd, z_percentage, min_points = 1024):
    """
    Crop a point cloud based on specified percentages of the range for the z dimension.

    Parameters:
    - pcd: Open3D point cloud object
    - z_percentage: Percentage of the z range to keep (value between 0 and 1)
    - negative: If True, keep points above the threshold, otherwise keep points below the threshold

    Returns:
    - index_groups: List of lists containing indices for each z range segment
    """
    # Get the points from the point cloud
    points = np.asarray(pcd.points)

    # Calculate the range for the z dimension
    z_min = np.min(points[:, 2])
    z_max = np.max(points[:, 2])
    z_range = z_max - z_min

    # Calculate the number of segments based on the percentage
    num_segments = int(1 / z_percentage)

    # Initialize the list to store index groups
    index_groups = []

    # Iterate over segments and calculate indices for each segment
    for i in tqdm(range(num_segments), f"Going through Z Segments ({z_percentage*100}..", ascii=True,  total=num_segments,dynamic_ncols=False):
        # Calculate thresholds for the current segment
        segment_min = z_min + (i * z_percentage * z_range)
        segment_max = z_min + ((i + 1) * z_percentage * z_range)

        # Create a mask for points within the current segment
        mask = np.where((points[:, 2] >= segment_min) & (points[:, 2] < segment_max))[0]
        if len(mask) >= min_points:
            # Add the indices to the index groups list
            index_groups.append(mask.tolist())

    return index_groups

def crop_group_point_cloud_by_percentage_y(pcd, y_percentage, min_points = 1024):
    """
    Crop a point cloud based on specified percentages of the range for the z dimension.

    Parameters:
    - pcd: Open3D point cloud object
    - z_percentage: Percentage of the z range to keep (value between 0 and 1)
    - negative: If True, keep points above the threshold, otherwise keep points below the threshold

    Returns:
    - index_groups: List of lists containing indices for each z range segment
    """
    # Get the points from the point cloud
    points = np.asarray(pcd.points)

    # Calculate the range for the z dimension
    y_min = np.min(points[:, 1])
    y_max = np.max(points[:, 1])
    y_range = y_max - y_min

    # Calculate the number of segments based on the percentage
    num_segments = int(1 / y_percentage)

    # Initialize the list to store index groups
    index_groups = []

    # Iterate over segments and calculate indices for each segment
    for i in tqdm(range(num_segments), f"Going through Y Segments ({y_percentage*100}..",  total=num_segments,ascii=True, dynamic_ncols=False):
        # Calculate thresholds for the current segment
        segment_min = y_min + (i * y_percentage * y_range)
        segment_max = y_min + ((i + 1) * y_percentage * y_range)


        # Create a mask for points within the current segment
        mask = np.where((points[:, 1] >= segment_min) & (points[:, 1] < segment_max))[0]


        if len(mask) >= min_points:
            # Add the indices to the index groups list
            index_groups.append(mask.tolist())

    return index_groups

def crop_group_point_cloud_by_percentage_x(pcd, x_percentage, min_points = 1024):
    """
    Crop a point cloud based on specified percentages of the range for the z dimension.

    Parameters:
    - pcd: Open3D point cloud object
    - z_percentage: Percentage of the z range to keep (value between 0 and 1)
    - negative: If True, keep points above the threshold, otherwise keep points below the threshold

    Returns:
    - index_groups: List of lists containing indices for each z range segment
    """
    # Get the points from the point cloud
    points = np.asarray(pcd.points)

    # Calculate the range for the z dimension
    x_min = np.min(points[:, 0])
    x_max = np.max(points[:, 0])
    x_range = x_max - x_min

    # Calculate the number of segments based on the percentage
    num_segments = int(1 / x_percentage)

    # Initialize the list to store index groups
    index_groups = []
    # Iterate over segments and calculate indices for each segment
    for i in tqdm(range(num_segments), desc=f"Going through X Segments ({x_percentage*100}..", total=num_segments, ascii=True, dynamic_ncols=False):
        # Calculate thresholds for the current segment
        segment_min = x_min + (i * x_percentage * x_range)
        segment_max = x_min + ((i + 1) * x_percentage * x_range)

        # Create a mask for points within the current segment
        mask = np.where((points[:, 0] >= segment_min) & (points[:, 0] < segment_max))[0]

        if len(mask) >= min_points:
        # Add the indices to the index groups list
            index_groups.append(mask.tolist())

    return index_groups

def select_group_point_cloud_by_batch_size_percentage(pcd, batch_size_percentage=0.1):
    """
    Crop a point cloud based on specified percentages of the range for each dimension.

    Parameters:
    - pcd: Open3D point cloud object
    - batch_size_percentage: Percentage of the total points to include in each batch (value between 0 and 1)
    - negative: Boolean indicating whether to select batches from the end of the point cloud

    Returns:
    - batched_indices: List of lists containing indices for each batch
    """
    # Get the points from the point cloud
    points = np.asarray(pcd.points)
    num_points = len(points)

    # Calculate the batch size based on the percentage of total points
    batch_size = int(num_points * batch_size_percentage)

    # Initialize an empty list to store batched indices
    batched_indices = []

    # Generate batches of indices
    for i in tqdm(range(0, num_points, batch_size), desc=f"Going through Batches ({batch_size_percentage*100}..", total=1/batch_size_percentage, ascii=True, dynamic_ncols=False):
        batch_indices = np.arange(i, min(i + batch_size, num_points))
        if len(batch_indices) >= 0.8 * batch_size:
            batched_indices.append(batch_indices)
    return batched_indices

def generate_grid_feats_dataset(base_dir, output_path_esf, output_path_legend):
    subset_kwargs = dict(k=8, r_mul=3.0, aabb_pad_mul=6.0, rho_k=8, rho_sample=8000)
    voxel_kwargs = dict(
        rho=None, voxel_scales=None, origin_mode="union_min",
        use_scan_dilation=True, dilation_iters=1, dilation_connectivity=6,
        dens_ratio_clip=3.0, compute_chamfer=False)

    save_counter = 0
    if os.path.exists(output_path_esf):
        with open(output_path_esf, 'r') as f:
            esf_dict = json.load(f)
    else:
        esf_dict = {}

    if os.path.exists(output_path_legend):
        with open(output_path_legend, 'r') as f:
            legend_dict = json.load(f)
    else:
        legend_dict = {}

    percentages = [95, 90, 85, 80, 75, 70, 65, 60, 58, 55, 50, 45, 40, 35, 30] #[95, 85, 75, 65, 55, 50, 40, 30, 10]
    augment_types = [
        (crop_group_point_cloud_by_percentage_x, "x"),
        (crop_group_point_cloud_by_percentage_y, "y"),
        (crop_group_point_cloud_by_percentage_z, "z"),
        (select_group_point_cloud_by_batch_size_percentage, "batch")
    ]

    for folderidx, folder in tqdm(enumerate(sorted(os.listdir(base_dir))), desc="Folder",  dynamic_ncols=True, total =len(sorted(os.listdir(base_dir)))):
        start_foldertimer = time.time()

        print(folder)
        folder_path = os.path.join(base_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        if folder not in esf_dict:
            esf_dict[folder] = {}

        existing_indices = set(map(int, esf_dict[folder].keys()))
        next_idx = max(existing_indices) + 1 if existing_indices else 0

        files = sorted([f for f in os.listdir(folder_path) if f.endswith('.ply')])

        for idx, file in tqdm(enumerate(files), desc="Files",  dynamic_ncols=True, total =len(files)):
            if str(next_idx) in esf_dict[folder]:
                next_idx += 1
                continue
            print(f"📥 Verarbeite: {file}")
            start_instancetimer = time.time()
            file_path = os.path.join(folder_path, file)
            pcd = o3d.io.read_point_cloud(file_path)
            #pcd = preprocess_pcd_for_esf(pcd)
            points = np.asarray(pcd.points)
            esf_dict[folder][idx] = {}

            # Original

            esf_dict[folder][idx]["filename"] = file
            grid_feat_100 = features_for_object(pcd_ref_o3d=pcd, pcd_scan_obj_o3d=pcd, voxel_kwargs=voxel_kwargs)
            grid_feat_jitter = features_for_object(pcd_ref_o3d=pcd, pcd_scan_obj_o3d=jitter_pcd(pcd), voxel_kwargs=voxel_kwargs)

            esf_dict[folder][idx][100] = [grid_feat_100, grid_feat_jitter]

            # Downsampling
            if len(points) > 50000:
                sampled = [
                    uniform_subsample_point_cloud(points= points,num_points_to_select = k) for k in [5000, 10000, 50000]
                ]
            elif len(points) > 10000:
                sampled = [
                    uniform_subsample_point_cloud(points= points,num_points_to_select = k) for k in [1000, 5000, 10000]
                ]
            elif len(points) > 5000:
                sampled = [
                    uniform_subsample_point_cloud(points= points,num_points_to_select = k) for k in [1000, 5000]
                ]
            else:
                sampled = [uniform_subsample_point_cloud(points= points, num_points_to_select =1000)]

            for pc in sampled:
                if pc.has_points():
                    esf_dict[folder][idx][100].append(features_for_object(pcd_ref_o3d=pcd, pcd_scan_obj_o3d=pc, voxel_kwargs=voxel_kwargs))
                    esf_dict[folder][idx][100].append(features_for_object(pcd_ref_o3d=pcd, pcd_scan_obj_o3d=jitter_pcd(pc), voxel_kwargs=voxel_kwargs))

            # Prozentsätze mit Crops und Jitter
            for perc in percentages:
                esf_list = []
                legend_entry = []
                pcd_jitter = jitter_pcd(pcd)
                for func, name in augment_types:
                    for p in [pcd, pcd_jitter]:
                        suffix = f"{name}_jitter" if p is pcd_jitter else name
                        idx_groups = func(copy.deepcopy(p), perc/100.0)

                        for group in idx_groups:
                            if len(group) == 0:
                                continue  # leere Gruppe überspringen
                            cropped = p.select_by_index(group)
                            if len(cropped.points) == 0:
                                continue

                            esf_list.append(features_for_object(pcd_ref_o3d=pcd, pcd_scan_obj_o3d=cropped,voxel_kwargs=voxel_kwargs))
                            legend_entry.append(suffix)

                esf_dict[folder][idx][perc] = esf_list
                legend_dict[perc] = legend_entry
            print("TIME FOR INSTANCE PROCESSING: ", time.time()-start_instancetimer)
            if save_counter % 10 == 0:
                with open(output_path_esf, 'w') as f:
                    json.dump(esf_dict, f)
                with open(output_path_legend, 'w') as f:
                    json.dump(legend_dict, f)

            next_idx += 1
            save_counter +=1

        print("TIME FOR FOLDER PROCESSING: ", time.time() - start_foldertimer)

    print(f"✅ ESF dataset saved to {output_path_esf}")
    print(f"✅ Legend saved to {output_path_legend}")


def voxel_downsample(pcd= None, points=None, bins= 64):
    if type(points) == type(None):
        points = np.array(pcd.points)
    voxel_size = 1.0 / bins
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    down = pcd.voxel_down_sample(voxel_size=voxel_size)
    return down, np.asarray(down.points)

def plot_radial_histogram(hist):
    plt.figure(figsize=(6,5))
    plt.imshow(hist, origin='lower', cmap='viridis', aspect='auto')
    plt.colorbar(label='Dichte')
    plt.xlabel('Theta-Bins (Richtung)')
    plt.ylabel('Radius-Bins')
    plt.title('Radiale Dichteverteilung')
    plt.show()

def plot_normal_histogram(hist):
    plt.figure(figsize=(6,5))
    plt.imshow(hist, origin='lower', cmap='viridis', aspect='auto')
    plt.colorbar(label='Häufigkeit')
    plt.xlabel('Azimut-Bins')
    plt.ylabel('Elevations-Bins')
    plt.title('Normalenrichtungsverteilung')
    plt.show()

def normals_distribution_histogram(points: np.ndarray, bins_az=8, bins_el=8, k_neighbors=30):
    # Normalen berechnen mit open3d
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    radius = calculate_mean_radius(pcd, k = k_neighbors)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
    normals = np.asarray(pcd.normals)

    # In sphärische Koordinaten
    nx, ny, nz = normals[:, 0], normals[:, 1], normals[:, 2]
    az = np.arctan2(ny, nx)  # -pi..pi
    el = np.arccos(np.clip(nz, -1.0, 1.0))  # 0..pi

    # Binning
    az_bins = np.linspace(-np.pi, np.pi, bins_az + 1)
    el_bins = np.linspace(0, np.pi, bins_el + 1)
    hist, _, _ = np.histogram2d(az, el, bins=[az_bins, el_bins])

    # Normalisierung
    hist = hist / (np.sum(hist) + 1e-8)
    return hist.astype(np.float32).tolist()  # shape (bins_az, bins_el)

def generate_normal_xray_hist(pcd =None, points = None):
    if points == None:
        points = np.array(pcd.points)


    pcd_down_normal, downsampled_normal = voxel_downsample(points=points, bins=128)
    normal_hist = normals_distribution_histogram(downsampled_normal, bins_az=8, bins_el=8)

    # Reihenweise flatten:
    hist_flat = np.array(normal_hist).flatten(order='C')
    print("Normal Hist Shape:", np.array(normal_hist).shape)

    return list(normal_hist), list(hist_flat)


import numpy as np
from scipy.spatial import cKDTree

def extract_scan_subset_for_ref(
    pcd_ref: np.ndarray,          # (Nr,3) float32
    pcd_scan_full: np.ndarray,    # (Ns,3) float32  (ganzer Scan)
    scan_normals: np.ndarray=None,# optional (Ns,3)
    k: int = 8,                   # k-NN pro Ref-Punkt
    r_mul: float = 3.0,           # Radius = r_mul * rho_local
    rho_k: int = 8,               # k für rho-Schätzung (scan-scan)
    rho_sample: int = 8000,       # Subsample für rho lokale Schätzung
    aabb_pad_mul: float = 6.0,    # AABB-Puffer = aabb_pad_mul * rho (wenn rho noch unbekannt: ~5% Diagonale)
    normal_consistency_deg: float = None,  # z.B. 35°; benötigt scan_normals
    max_neighbors_per_ref: int = None,     # optional hartes Limit (sonst <=k mit Radiusfilter)
    return_indices: bool = False
):
    """
    Liefert:
      pcd_scan_obj ........ Scan-Untermenge passend zum Referenzobjekt
      rho_local ............ lokale Scandichte (Median-Abstand k=rho_k)
      idx_global (opt) ..... Indizes in pcd_scan_full
    Vorgehen:
      1) lokale Scanregion per AABB(ref) + Puffer vorfiltern
      2) rho_local auf der lokalen Scanregion schätzen (scan-scan)
      3) k-NN von ref -> lokale Scanregion, mit distance_upper_bound = r_mul*rho_local
      4) optionale Normal-Konsistenz filtern
      5) eindeutige Scan-Indizes sammeln
    """
    pcd_ref = np.asarray(pcd_ref, dtype=np.float32)
    pcd_scan_full = np.asarray(pcd_scan_full, dtype=np.float32)
    assert pcd_ref.ndim==2 and pcd_ref.shape[1]==3
    assert pcd_scan_full.ndim==2 and pcd_scan_full.shape[1]==3

    if len(pcd_scan_full) == 0:
        out = (pcd_scan_full.copy(), 1e-3)
        return (*out, np.empty((0,), np.int64)) if return_indices else out

    # --- 1) Grobregion um Referenz bestimmen (AABB + Puffer) ---
    ref_min = pcd_ref.min(0); ref_max = pcd_ref.max(0)
    diag = np.linalg.norm(ref_max - ref_min)
    # grober Puffer, falls rho noch unbekannt
    pad0 = 0.05 * diag if np.isfinite(diag) and diag>0 else 0.1

    # grob vorfiltern (AABB + pad0)
    smin0 = ref_min - pad0
    smax0 = ref_max + pad0
    m0 = np.all((pcd_scan_full >= smin0) & (pcd_scan_full <= smax0), axis=1)
    scan_local0 = pcd_scan_full[m0]
    idx_local0 = np.nonzero(m0)[0]

    if len(scan_local0) == 0:
        out = (scan_local0, 1e-3)
        return (*out, idx_local0) if return_indices else out

    # --- 2) rho_local auf lokaler Scanregion schätzen (scan-scan) ---
    # subsample
    n_samp = min(rho_sample, len(scan_local0))
    samp_idx = np.random.choice(len(scan_local0), size=n_samp, replace=False)
    samp = scan_local0[samp_idx]
    tree_local = cKDTree(scan_local0)
    dists, _ = tree_local.query(samp, k=min(rho_k+1, len(scan_local0)))
    dk = dists[:, -1] if dists.ndim==2 else dists
    rho_local = float(np.median(dk)) if np.isfinite(dk).all() and np.median(dk)>0 else 1e-3

    # verfeinerter Puffer mit rho
    pad = aabb_pad_mul * rho_local
    smin = ref_min - pad
    smax = ref_max + pad
    m = np.all((pcd_scan_full >= smin) & (pcd_scan_full <= smax), axis=1)
    scan_local = pcd_scan_full[m]
    idx_local = np.nonzero(m)[0]

    if len(scan_local) == 0:
        out = (scan_local, rho_local)
        return (*out, idx_local) if return_indices else out

    # lokalen Baum für KNN
    tree = cKDTree(scan_local)

    # --- 3) k-NN (mit Radius-Cutoff) von Ref-Punkten zur lokalen Region ---
    radius = r_mul * rho_local
    k_eff = min(k, len(scan_local))
    d, j = tree.query(pcd_ref, k=k_eff, distance_upper_bound=radius)

    # in den Fällen k_eff==1, query gibt 1D-Arrays zurück -> in 2D heben
    if k_eff == 1:
        d = d[:, None]
        j = j[:, None]

    # gültige Treffer: j < len(scan_local) (sonst == len -> "kein Nachbar")
    valid = j < len(scan_local)
    if normal_consistency_deg is not None and scan_normals is not None:
        # optionale Normalprüfung (nur wenn du zu ref Normale hast; hier skippen wir Ref-Normale -> nur Scan-Normale vorhanden?)
        # Falls ref-Normalen existieren, reiche sie hier rein und prüfe Winkel.
        pass  # Platzhalter – falls du ref-Normalen hast, kann ich dir die Zeilen ergänzen.

    # optional begrenzen pro Ref
    if max_neighbors_per_ref is not None and max_neighbors_per_ref < k_eff:
        # wähle pro Ref die kleinsten Distanzen
        order = np.argsort(d, axis=1)
        keep_cols = order[:, :max_neighbors_per_ref]
        row_idx = np.arange(d.shape[0])[:, None]
        mask_small = np.zeros_like(valid, dtype=bool)
        mask_small[row_idx, keep_cols] = True
        valid = valid & mask_small

    sel = j[valid].ravel()
    sel = sel[sel < len(scan_local)]
    if sel.size == 0:
        out = (scan_local[:0], rho_local)
        return (*out, idx_local[:0]) if return_indices else out

    # --- 4) eindeutige Scan-Indizes einsammeln ---
    sel_unique = np.unique(sel)
    pcd_scan_obj = scan_local[sel_unique]
    idx_global = idx_local[sel_unique]  # zurück auf globalen Scan gemappt

    return (pcd_scan_obj, rho_local, idx_global) if return_indices else (pcd_scan_obj, rho_local)

# --- rho-Helfer ---
def _estimate_rho_from_scan(scan: np.ndarray, k: int, n_samp: int) -> float:
    if scan.size == 0:
        return 1e-3
    ns = min(n_samp, len(scan))
    idx = np.random.choice(len(scan), size=ns, replace=False)
    samp = scan[idx]
    tree = cKDTree(scan)
    kk = min(k+1, len(scan))
    d, _ = tree.query(samp, k=kk)
    dk = d[:, -1] if d.ndim == 2 else d
    med = np.median(dk)
    return float(med if np.isfinite(med) and med>0 else 1e-3)
def compute_voxel_features(
    pcd_ref: np.ndarray,                 # (Nr,3) float32
    pcd_scan: np.ndarray,                # (Ns,3) float32 (bereits auf Objekt beschnitten)
    rho: Optional[float] = None,         # wenn None/False -> aus Scan schätzen
    rho_k: int = 8,                      # k für rho-Schätzung (scan-scan)
    rho_sample: int = 8000,              # subsample für rho
    voxel_scales: Optional[List[float]] = None,  # z.B. [6*rho, 3*rho, 1.5*rho]
    origin_mode: str = "union_min",      # "union_min" | "ref_min" | "scan_min" | "zero"
    use_scan_dilation: bool = True,      # toleranter Abgleich (gegen kleine Shifts)
    dilation_iters: int = 1,
    dilation_connectivity: int = 6,      # 6 oder 26
    dens_ratio_clip: float = 3.0,        # Clip für Dichteverhältnis
    compute_chamfer: bool = False,       # optional (langsamer)
    chamfer_cap_mul: float = 3.0,        # Cap-Radius = chamfer_cap_mul * s2
    chamfer_sample: int = 20000,         # max Punkte pro Richtung für sCD
    return_debug: bool = False
) -> Dict[str, Any]:
    """
    Berechnet robuste, schnelle Multi-Scale-Features für Ref/Scan auf gemeinsamem Voxelgitter.
    - Wenn rho nicht gegeben: Schätzung aus pcd_scan (Median-Abstand zum k=rho_k-Nachbarn).
    - Voxel-Pyramide (Default): s0=6*rho, s1=3*rho, s2=1.5*rho
    - Gemeinsamer Origin je Level: floor(min_xyz(Ref ∪ Scan)/s)*s
    Features:
      * Coverage_R/S, IoU je Level
      * Occupancy-Volumenverhältnis (#ScanVox/#RefVox) je Level
      * AABB-Volumenverhältnis (weltbasiert) je Level
      * Dichteverhältnis (Achsen-Histogramme) auf feinster Stufe
      * optional: gecappte symmetrische Chamferdistanz (sCD_cap)
    """
    def _safe_array(x):
        a = np.asarray(x, dtype=np.float32)
        return a if (a.ndim==2 and a.shape[1]==3) else a.reshape(-1, 3).astype(np.float32)

    pcd_ref  = _safe_array(pcd_ref)
    pcd_scan = _safe_array(pcd_scan)

    out: Dict[str, Any] = {"features": {}}

    # --- frühe Abbrüche ---
    if pcd_ref.size == 0 and pcd_scan.size == 0:
        out["rho"] = float(rho) if rho else 1e-3
        out["scales"] = voxel_scales or []
        out["features"].update({
            "empty": True, "cov_R_s0": 0.0, "cov_S_s0": 0.0, "iou_s0": 0.0
        })
        return out
    if pcd_ref.size == 0:
        # kein Ref -> alle Coverage-Features 0; dennoch rho bestimmen, falls nötig
        if not rho:
            rho = _estimate_rho_from_scan(pcd_scan, rho_k, rho_sample)
        s_list = voxel_scales or [6.0*float(rho), 3.0*float(rho), 1.5*float(rho)]
        out["rho"] = float(rho); out["scales"] = list(map(float, s_list))
        for li, s in enumerate(s_list):
            out["features"][f"cov_R_s{li}"] = 0.0
            out["features"][f"cov_S_s{li}"] = 0.0
            out["features"][f"iou_s{li}"]   = 0.0
            out["features"][f"occ_vol_ratio_s{li}"] = 0.0
            out["features"][f"aabb_vol_ratio_s{li}"] = 0.0
        return out
    if pcd_scan.size == 0:
        # kein Scan -> Coverage_R=0, IoU=0, Verhältnisse=0
        if not rho:
            rho = 1e-3
        s_list = voxel_scales or [6.0*float(rho), 3.0*float(rho), 1.5*float(rho)]
        out["rho"] = float(rho); out["scales"] = list(map(float, s_list))
        for li, s in enumerate(s_list):
            out["features"][f"cov_R_s{li}"] = 0.0
            out["features"][f"cov_S_s{li}"] = 0.0
            out["features"][f"iou_s{li}"]   = 0.0
            out["features"][f"occ_vol_ratio_s{li}"] = 0.0
            out["features"][f"aabb_vol_ratio_s{li}"] = 0.0
        return out

    # --- rho bestimmen (falls nicht gegeben) ---
    if not rho:
        rho = _estimate_rho_from_scan(pcd_scan, rho_k, rho_sample)
    out["rho"] = float(rho)

    # --- Skalenliste festlegen ---
    if voxel_scales is None:
        s_list = [6.0*rho, 3.0*rho, 1.5*rho]
    else:
        s_list = list(map(float, voxel_scales))
    out["scales"] = list(map(float, s_list))

    # --- Hilfsfunktionen ---
    def _origin_for_scale(s: float) -> np.ndarray:
        if origin_mode == "ref_min":
            mn = pcd_ref.min(0)
        elif origin_mode == "scan_min":
            mn = pcd_scan.min(0)
        elif origin_mode == "zero":
            mn = np.zeros(3, dtype=np.float32)
        else:  # union_min
            mn = np.minimum(pcd_ref.min(0), pcd_scan.min(0))
        return np.floor(mn / s) * s

    def _voxelize(points: np.ndarray, s: float, o: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return np.empty((0, 3), dtype=np.int32)
        idx = np.floor((points - o) / s).astype(np.int32)
        if idx.size == 0:
            return idx
        # unique Voxel
        # (np.unique axis=0 ist schnell genug für typ. Objektgrößen)
        uv = np.unique(idx, axis=0)
        return uv

    def _dilate(vox: np.ndarray, conn: int, iters: int) -> np.ndarray:
        if vox.size == 0 or iters <= 0:
            return vox
        if conn == 26:
            offs = np.array([[dx,dy,dz] for dx in (-1,0,1)
                                        for dy in (-1,0,1)
                                        for dz in (-1,0,1)
                                        if not (dx==0 and dy==0 and dz==0)], dtype=np.int32)
        else:  # 6er Nachbarschaft
            offs = np.array([[ 1,0,0],[-1,0,0],[0, 1,0],[0,-1,0],[0,0, 1],[0,0,-1]], dtype=np.int32)
        cur = vox
        for _ in range(iters):
            neigh = (cur[:,None,:] + offs[None,:,:]).reshape(-1, 3)
            cur = np.unique(np.vstack([cur, neigh]), axis=0)
        return cur

    def _aabb_volume_world(vox: np.ndarray, s: float) -> float:
        if vox.size == 0:
            return 0.0
        vmin = vox.min(0); vmax = vox.max(0)
        # +1 weil Voxel-Indizes inklusiv sind
        ext = (vmax - vmin + 1).astype(np.float32) * float(s)
        vol = float(ext[0]*ext[1]*ext[2])
        return vol

    def _axis_density_ratio(ref_vox: np.ndarray, scan_vox: np.ndarray) -> Dict[str, float]:
        # nur auf feinster Stufe sinnvoll; Verhältnisse je Achse
        feats = {}
        if ref_vox.size == 0:
            for ax in "xyz":
                feats[f"dens_ratio_mean_{ax}"] = 0.0
                feats[f"dens_ratio_med_{ax}"]  = 0.0
                feats[f"dens_ratio_p10_{ax}"]  = 0.0
                feats[f"dens_ratio_p90_{ax}"]  = 0.0
            return feats

        for axis, axname in enumerate(["x","y","z"]):
            r_vals, r_cnts = np.unique(ref_vox[:,axis], return_counts=True)
            s_vals, s_cnts = np.unique(scan_vox[:,axis], return_counts=True) if scan_vox.size else (np.array([],dtype=np.int32), np.array([],dtype=np.int64))
            s_map = dict(zip(s_vals.tolist(), s_cnts.tolist()))
            # Ratio nur dort, wo Ref belegt ist; Clip gegen Ausreißer
            ratios = []
            for v, rc in zip(r_vals.tolist(), r_cnts.tolist()):
                sc = s_map.get(v, 0)
                if rc > 0:
                    ratios.append(min(dens_ratio_clip, sc/rc))
            if len(ratios) == 0:
                stats = (0.0, 0.0, 0.0, 0.0)
            else:
                arr = np.asarray(ratios, dtype=np.float32)
                stats = (float(arr.mean()),
                         float(np.median(arr)),
                         float(np.percentile(arr, 10)),
                         float(np.percentile(arr, 90)))
            feats[f"dens_ratio_mean_{axname}"] = stats[0]
            feats[f"dens_ratio_med_{axname}"]  = stats[1]
            feats[f"dens_ratio_p10_{axname}"]  = stats[2]
            feats[f"dens_ratio_p90_{axname}"]  = stats[3]
        return feats

    def _set_metrics(ref_vox: np.ndarray, scan_vox: np.ndarray) -> Tuple[float,float,float,int,int,int]:
        # Coverage/IoU auf Voxel-Sets
        if ref_vox.size == 0 and scan_vox.size == 0:
            return 0.0, 0.0, 0.0, 0, 0, 0
        ref_set  = {tuple(p) for p in ref_vox} if ref_vox.size else set()
        scan_set = {tuple(p) for p in scan_vox} if scan_vox.size else set()
        inter = len(ref_set & scan_set)
        nR = len(ref_set); nS = len(scan_set)
        uni = nR + nS - inter if (nR or nS) else 0
        cov_R = inter / nR if nR>0 else 0.0
        cov_S = inter / nS if nS>0 else 0.0
        iou   = inter / uni if uni>0 else 0.0
        return cov_R, cov_S, iou, inter, nR, nS



    # --- Multi-Scale Pipeline ---
    debug_info = {"levels": []} if return_debug else None

    finest_ref_vox = None
    finest_scan_vox = None

    for li, s in enumerate(s_list):
        o = _origin_for_scale(s)
        ref_vox  = _voxelize(pcd_ref,  s, o)
        scan_vox = _voxelize(pcd_scan, s, o)

        # toleranz-Dilatation nur für Scan (Reduktion FN)
        if use_scan_dilation and scan_vox.size:
            scan_vox = _dilate(scan_vox, dilation_connectivity, dilation_iters)

        cov_R, cov_S, iou, inter, nR, nS = _set_metrics(ref_vox, scan_vox)
        out["features"][f"cov_R_s{li}"] = float(cov_R)
        out["features"][f"cov_S_s{li}"] = float(cov_S)
        out["features"][f"iou_s{li}"]   = float(iou)

        # Occupancy-"Volumen"-Verhältnis (über Anzahlen belegter Voxel)
        occ_ratio = (nS / nR) if nR>0 else 0.0
        out["features"][f"occ_vol_ratio_s{li}"] = float(occ_ratio)

        # AABB-Volumenverhältnis in Weltkoordinaten
        v_ref = _aabb_volume_world(ref_vox, s)
        v_scn = _aabb_volume_world(scan_vox, s)
        aabb_ratio = (v_scn / v_ref) if v_ref>0 else 0.0
        out["features"][f"aabb_vol_ratio_s{li}"] = float(aabb_ratio)

        if li == len(s_list)-1:
            finest_ref_vox  = ref_vox
            finest_scan_vox = scan_vox

        if return_debug:
            debug_info["levels"].append({
                "scale": float(s),
                "origin": o.tolist(),
                "n_ref_vox": int(nR),
                "n_scan_vox": int(nS),
                "n_inter": int(inter)
            })

    # --- Dichteverhältnis (Achsen) auf feinster Stufe ---
    if finest_ref_vox is None:
        # Falls keine Schleife o.Ä. – sollte nicht passieren, aber safety:
        o = _origin_for_scale(s_list[-1])
        finest_ref_vox  = _voxelize(pcd_ref,  s_list[-1], o)
        finest_scan_vox = _voxelize(pcd_scan, s_list[-1], o)
        if use_scan_dilation and finest_scan_vox.size:
            finest_scan_vox = _dilate(finest_scan_vox, dilation_connectivity, dilation_iters)

    dens_feats = _axis_density_ratio(finest_ref_vox, finest_scan_vox)
    out["features"].update(dens_feats)

    # --- optional: gecappte symmetrische Chamferdistanz ---
    if compute_chamfer:
        s2 = s_list[-1]
        cap_r = chamfer_cap_mul * s2
        scd = _symmetric_capped_chamfer(pcd_ref, pcd_scan, cap_r, chamfer_sample)
        out["features"]["sCD_cap"] = float(scd)
        out["tolerances"] = {"cap_radius": float(cap_r),
                             "dilation": int(dilation_iters) if use_scan_dilation else 0,
                             "conn": int(dilation_connectivity)}
    else:
        out["tolerances"] = {"dilation": int(dilation_iters) if use_scan_dilation else 0,
                             "conn": int(dilation_connectivity)}

    if return_debug:
        out["debug"] = debug_info
    return out


def radial_density_histogram(points: np.ndarray, r_bins=8, theta_bins=8):
    # Punkte in sphärische Koordinaten
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r = np.linalg.norm(points, axis=1)
    theta = np.arccos(np.clip(z / (r + 1e-8), -1.0, 1.0))  # polarwinkel 0..pi

    # Binning
    r_hist_bins = np.linspace(0.0, 1.0, r_bins + 1)
    t_hist_bins = np.linspace(0.0, np.pi, theta_bins + 1)
    hist, _, _ = np.histogram2d(r, theta, bins=[r_hist_bins, t_hist_bins])

    # Normalisierung
    hist = hist / (np.sum(hist) + 1e-8)
    return hist.astype(np.float32)  # shape (r_bins, theta_bins)

def calculate_mean_radius(pcd, k = 10):
    # Annahme: points ist eine numpy-Array mit den Koordinaten der Punktwolke
    # Beispiel: points = np.array([[x1, y1, z1], [x2, y2, z2], ...])

    # Definieren der Anzahl von Nachbarn (k)

    # Initialisieren der NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k)

    # Anpassen der Punktwolke an die NearestNeighbors
    nn.fit(np.array(pcd.points))

    # Berechnen der Entfernungen und Indizes der k-nächsten Nachbarn für jeden Punkt
    distances, indices = nn.kneighbors(np.array(pcd.points))

    # Berechnung des durchschnittlichen Abstands für jeden Punkt
    mean_distances = np.mean(distances[:, 1:-1], axis=1)

    # Berechnung des Gesamtdurchschnitts
    overall_mean_distance = np.round(np.mean(mean_distances),3)

    # Ausgabe des Gesamtdurchschnitts
    print("Overall mean distance:", overall_mean_distance)
    return overall_mean_distance

def features_for_object(
    pcd_ref_o3d,
    pcd_scan_full_o3d=None,          # kompletter Scan (falls Subset nötig)
    pcd_scan_obj_o3d=None,           # oder bereits extrahiertes Objekt-Subset
    subset_kwargs=None,              # Args für extract_scan_subset_for_ref
    voxel_kwargs=None,               # Args für compute_voxel_features
    return_meta: bool = False,
    force_print_order: bool = False  # Erzwingt erneutes Ausgeben der Reihenfolge
):
    """
    Liefert standardmäßig NUR den flachen Feature-Vektor (list[float]).
    Setze return_meta=True, um zusätzlich {'keys', 'rho', 'scales', 'idx_global'} zu bekommen.
    Die Feature-Reihenfolge wird beim ersten Aufruf (oder wenn force_print_order=True) geloggt.

    Reihenfolge (für L=len(scales), typ. L=3):
      pro Level s0..s{L-1}:
        cov_R_sℓ, cov_S_sℓ, iou_sℓ, occ_vol_ratio_sℓ, aabb_vol_ratio_sℓ
      danach (feinste Stufe):
        dens_ratio_mean_x/y/z, dens_ratio_med_x/y/z,
        dens_ratio_p10_x/y/z,  dens_ratio_p90_x/y/z
    """
    import numpy as np

    subset_kwargs = subset_kwargs or {}
    voxel_kwargs  = (voxel_kwargs or {}).copy()

    ref = np.asarray(pcd_ref_o3d.points, dtype=np.float32)
    idx_global = None
    rho_hint = voxel_kwargs.pop("rho", None)

    # 1) Scan-Objekt bestimmen (falls nötig) + rho-Hint
    if pcd_scan_obj_o3d is None:
        assert pcd_scan_full_o3d is not None, "Entweder pcd_scan_obj_o3d ODER pcd_scan_full_o3d angeben."
        scan_full = np.asarray(pcd_scan_full_o3d.points, dtype=np.float32)
        scan_obj, rho_local, idx_global = extract_scan_subset_for_ref(
            ref, scan_full, return_indices=True, **subset_kwargs
        )
        if rho_hint is None:
            rho_hint = rho_local
    else:
        scan_obj = np.asarray(pcd_scan_obj_o3d.points, dtype=np.float32)

    # 2) Voxel-Features
    out = compute_voxel_features(
        pcd_ref=ref,
        pcd_scan=scan_obj,
        rho=rho_hint,
        **voxel_kwargs
    )

    # 3) Flatten in stabiler Reihenfolge
    feats = out.get("features", {})
    scales = out.get("scales", [])
    L = len(scales) if len(scales) > 0 else 3

    keys = []
    for li in range(L):
        keys += [
            f"cov_R_s{li}", f"cov_S_s{li}", f"iou_s{li}",
            f"occ_vol_ratio_s{li}", f"aabb_vol_ratio_s{li}"
        ]
    for ax in ("x","y","z"):
        keys += [
            f"dens_ratio_mean_{ax}", f"dens_ratio_med_{ax}",
            f"dens_ratio_p10_{ax}",  f"dens_ratio_p90_{ax}"
        ]

    vec = [float(feats.get(k, 0.0)) for k in keys]

    # 4) Reihenfolge 1x loggen
    if force_print_order or not getattr(features_for_object, "_printed_order", False):
        print("[features_for_object] Feature order:")
        for i, k in enumerate(keys):
            print(f"  {i:02d}: {k}")
        features_for_object._printed_order = True

    if return_meta:
        meta = {
            "keys": keys,
            "rho": float(out.get("rho", 0.0)),
            "scales": scales,
        }
        if idx_global is not None:
            meta["idx_global"] = idx_global.tolist()
        return vec, meta

    return vec


AUGMENT_FUNCS = [
    (crop_group_point_cloud_by_percentage_x, "x"),
    (crop_group_point_cloud_by_percentage_y, "y"),
    (crop_group_point_cloud_by_percentage_z, "z"),
    (select_group_point_cloud_by_batch_size_percentage, "batch"),
]

# ---- Default-Parameter ----
DEFAULT_SHARD_SIZE = 200_000
DEFAULT_FLUSH_EVERY = 10_000

# ---- Voxel-Feature-Parameter (wie in deiner Quelle) ----
SUBSET_KW = dict(k=8, r_mul=3.0, aabb_pad_mul=6.0, rho_k=8, rho_sample=8000)
VOXEL_KW = dict(
    rho=None, voxel_scales=None, origin_mode="union_min",
    use_scan_dilation=True, dilation_iters=1, dilation_connectivity=6,
    dens_ratio_clip=3.0, compute_chamfer=False
)

# ----------------------------- Utility
def parse_key(key_str: str) -> Tuple[str,str,str,int]:
    """
    Erwartet z.B. 'class_with_underscores_007_55_12'
    -> (cls, inst, perc, idx)
    """
    parts = key_str.split("_")
    cls = "_".join(parts[:-3])
    inst = parts[-3]
    perc = parts[-2]
    idx = int(parts[-1])
    return cls, inst, perc, idx

def state_paths(out_dir: str, fold: str, split: str):
    base = os.path.join(out_dir, f"{fold}-{split}")
    return base + ".state.json", base + ".state.json.tmp"

def shard_path(out_dir: str, fold: str, split: str, shard_id: int, tmp: bool = False) -> str:
    p = os.path.join(out_dir, f"{fold}-{split}.part{shard_id:03d}.jsonl")
    return p + ".tmp" if tmp else p

def load_state(out_dir: str, fold: str, split: str) -> Dict[str, Any]:
    path, _ = state_paths(out_dir, fold, split)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"idx": 0, "shard_id": 0, "in_shard": 0, "written_total": 0, "skipped_total": 0}

def save_state(out_dir: str, fold: str, split: str, st: Dict[str, Any]):
    path, tmp = state_paths(out_dir, fold, split)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f)
    os.replace(tmp, path)

# ----------------------------- Grid-JSON Zugriff
def grid_get_file(grid: Dict[str, Any], cls: str, inst: str) -> str:
    """
    Liefert den Dateinamen (PLY) für Klassenordner 'cls' und Index 'inst'.
    """
    entry = grid.get(cls, {}).get(int(inst), {})
    fname = entry.get("filename", None)
    if fname is None:
        # Fallback: manche Dumps speichern 'inst' als str
        entry = grid.get(cls, {}).get(str(inst), {})
        fname = entry.get("filename", None)
    return fname

def grid_get_vec(grid: Dict[str, Any], cls: str, inst: str, perc: str, idx: int):
    """
    Liefert vorab berechneten 27D-Featurevektor aus dem Grid-JSON.
    """
    Gc = grid.get(cls, {})
    Gi = Gc.get(inst) if inst in Gc else Gc.get(int(inst), {})
    vecs = Gi.get(str(perc), None)
    if isinstance(vecs, list) and 0 <= idx < len(vecs):
        return vecs[idx]
    return None

# ----------------------------- Recompute-Pfad
def load_pcd(base_dir: str, cls: str, fname: str) -> o3d.geometry.PointCloud:
    p = os.path.join(base_dir, cls, fname)
    return o3d.io.read_point_cloud(p)

def reproduce_crop_groups(pcd: o3d.geometry.PointCloud, perc_int: int) -> List[List[int]]:
    """
    Baut die gleiche Gruppenliste wie beim Datengenerator.
    Reihenfolge: x, y, z, batch; jeweils ohne Jitter, dann mit Jitter.
    Gibt Indexlisten (Indices ins jeweilige pcd) zurück.
    """
    groups: List[List[int]] = []
    # ohne Jitter
    for func, _ in AUGMENT_FUNCS:
        groups.extend(func(pcd, perc_int/100.0))
    # mit Jitter
    p_j = jitter_pcd(pcd)
    for func, _ in AUGMENT_FUNCS:
        groups.extend(func(p_j, perc_int/100.0))
    return groups

def recompute_features_for_pair(base_dir: str,
                                grid: Dict[str, Any],
                                ref_cls: str, ref_inst: str,
                                scan_cls: str, scan_inst: str,
                                perc: str, idx: int,
                                rounding: int) -> List[float]:
    """
    Rekonstruiert Ref (100%) und Scan (laut perc/idx) und berechnet 27D-Features.
    """
    # Ref laden
    ref_fname = grid_get_file(grid, ref_cls, ref_inst)
    if ref_fname is None:
        return None
    pcd_ref = load_pcd(base_dir, ref_cls, ref_fname)

    # Scan laden
    scan_fname = grid_get_file(grid, scan_cls, scan_inst)
    if scan_fname is None:
        return None
    pcd_scan_full = load_pcd(base_dir, scan_cls, scan_fname)

    perc_int = int(perc)
    if perc_int >= 100:
        pcd_scan_obj = pcd_scan_full
    else:
        groups = reproduce_crop_groups(pcd_scan_full, perc_int)
        if idx < 0 or idx >= len(groups):
            return None
        sel = groups[idx]
        if len(sel) == 0:
            return None
        pcd_scan_obj = pcd_scan_full.select_by_index(sel)

    vec = features_for_object(
        pcd_ref_o3d=pcd_ref,
        pcd_scan_obj_o3d=pcd_scan_obj,
        subset_kwargs=SUBSET_KW,
        voxel_kwargs=VOXEL_KW,
        return_meta=False
    )
    if vec is None:
        return None
    return [round(float(x), rounding) for x in vec]

# ----------------------------- Verarbeitung eines Splits
def process_split(cv_pairs: List[Dict[str, Any]],
                  grid: Dict[str, Any],
                  base_dir: str,
                  out_dir: str,
                  fold: str, split: str,
                  shard_size: int,
                  flush_every: int,
                  rounding: int):

    os.makedirs(out_dir, exist_ok=True)
    st = load_state(out_dir, fold, split)

    written_this_run = 0
    skipped_this_run = 0

    # aktueller Shard im TMP-Append-Modus
    tmp_path = shard_path(out_dir, fold, split, st["shard_id"], tmp=True)
    fp = open(tmp_path, "ab")

    def rotate_shard():
        nonlocal fp
        fp.flush(); fp.close()
        final_path = shard_path(out_dir, fold, split, st["shard_id"], tmp=False)
        os.replace(tmp_path, final_path)  # atomar
        st["shard_id"] += 1
        st["in_shard"] = 0
        # neuen TMP anlegen
        new_tmp = shard_path(out_dir, fold, split, st["shard_id"], tmp=True)
        return open(new_tmp, "ab")

    try:
        rng = range(st["idx"], len(cv_pairs))
        pbar = tqdm(rng, desc=f"{fold}-{split}", initial=st["idx"], total=len(cv_pairs))
        for i in pbar:
            pair = cv_pairs[i]
            y = int(pair.get("label", 0))
            # Keys parsen
            r_cls, r_inst, r_perc, r_idx = parse_key(pair["esf_ref"])
            s_cls, s_inst, s_perc, s_idx = parse_key(pair["esf_scan"])

            # Featurequelle wählen
            if y == 1:
                # Vorberechnete Grid-Features verwenden (Scan-Key maßgeblich)
                feats = grid_get_vec(grid, s_cls, s_inst, s_perc, s_idx)
                feats = [round(float(x), rounding) for x in feats] if feats is not None else None
            else:
                # Neu berechnen, Ref=100 %, Scan gemäß perc/index
                feats = recompute_features_for_pair(
                    base_dir, grid,
                    r_cls, r_inst, s_cls, s_inst, s_perc, s_idx, rounding
                )

            if feats is None:
                st["skipped_total"] += 1
                skipped_this_run += 1
            else:
                rec = {
                    "x": feats,
                    "y": y,
                    "pair": {
                        "esf_ref": pair["esf_ref"],
                        "esf_scan": pair["esf_scan"]
                    },
                    "meta": {"fold": fold, "split": split,
                             "shard": st["shard_id"], "line": st["in_shard"]}
                }
                fp.write(orjson.dumps(rec)); fp.write(b"\n")
                st["in_shard"] += 1
                st["written_total"] += 1
                written_this_run += 1

            # Shard rotation
            if st["in_shard"] >= shard_size:
                fp = rotate_shard()

            # periodischer State-Flush
            if ((i + 1) % flush_every) == 0:
                st["idx"] = i + 1
                save_state(out_dir, fold, split, st)
                fp.flush()

        # finaler Flush + State
        st["idx"] = len(cv_pairs)
        save_state(out_dir, fold, split, st)
        fp.flush(); fp.close()

        # falls letzter Shard nicht leer -> finalisieren
        final_last = shard_path(out_dir, fold, split, st["shard_id"], tmp=False)
        if st["in_shard"] > 0 and os.path.exists(tmp_path):
            os.replace(tmp_path, final_last)

    except Exception as e:
        # Notfall-Flush des aktuellen TMP
        try:
            fp.flush(); fp.close()
        except Exception:
            pass
        print("⚠️ Abbruch. Buffer als .tmp belassen.")
        traceback.print_exc()

    print(f"✅ {fold}-{split}: geschrieben={written_this_run}, übersprungen={skipped_this_run}, shard={st['shard_id']:03d}, in_shard={st['in_shard']}")
    return written_this_run, skipped_this_run

# ----------------------------- Main über alle Folds
def process_all(cv: Dict[str, Any],
                grid: Dict[str, Any],
                base_dir: str,
                out_dir: str,
                shard_size: int,
                flush_every: int,
                rounding: int):

    total_w = total_s = 0
    for fold_name, fold_dict in cv.items():
        for split in ("train", "val", "test"):
            pairs = fold_dict.get(split, [])
            w, s = process_split(
                cv_pairs=pairs, grid=grid, base_dir=base_dir,
                out_dir=out_dir, fold=fold_name, split=split,
                shard_size=shard_size, flush_every=flush_every,
                rounding=rounding
            )
            total_w += w; total_s += s
    print(f"🏁 Gesamt: geschrieben={total_w}, übersprungen={total_s}")

def main():
    dir_script = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser("Grid-Feature-Sharder (JSONL, resumable, atomic).")
    ap.add_argument("--cv_info", default=os.path.join(dir_script, "cv6_info.json"), help="Pfad zu CV-Info JSON")
    ap.add_argument("--grid_json", default=os.path.join(dir_script, "verf_grid_check_hist_dataset3_newclasses.json"),help="Pfad zum vorab erzeugten Grid-JSON")
    ap.add_argument("--base_dir", default=".", help="Root der Klassenordner mit .ply-Dateien")
    ap.add_argument("--out_dir", default=os.path.join(dir_script,"features_stream"), help="Ausgabeverzeichnis")
    ap.add_argument("--shard_size", type=int, default=DEFAULT_SHARD_SIZE)
    ap.add_argument("--flush_every", type=int, default=DEFAULT_FLUSH_EVERY)
    ap.add_argument("--round", type=int, default=4, help="Dezimalstellen für Rundung")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.cv_info, "r", encoding="utf-8") as f:
        cv = json.load(f)
    with open(args.grid_json, "r", encoding="utf-8") as f:
        grid = json.load(f)

    process_all(cv=cv, grid=grid, base_dir=args.base_dir, out_dir=args.out_dir,
                shard_size=args.shard_size, flush_every=args.flush_every,
                rounding=args.round)

if __name__ == "__main__":
    main()
