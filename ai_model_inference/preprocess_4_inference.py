from typing import Dict, Tuple, Optional, List, Any
from torch.utils.data import Dataset, DataLoader
import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree
import os
import json
from tqdm import tqdm
import time
import copy
from sklearn.neighbors import KDTree
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from scipy.stats import wasserstein_distance
import threading
import subprocess
##################################ESF DIFF DESCRIPTOR###################################
# functions for generating normal hist features:
def compute_esf_descriptor(pcd_o3d=None, points = None, exe_path = os.path.join(os.getcwd(), "esf_estimation.exe"), use_normals = False):
    if type(points) == type(None):
        # Save the point cloud as a PCD file
        pcd_o3d, normal_vectors = calculate_normals(pcd_o3d, k=30, radius=None)
        o3d.io.write_point_cloud("esf_in_pcd_temp.pcd", pcd_o3d)
        if np.array(pcd_o3d.points).shape[0]> 1000000:
            time.sleep(0.5)
    elif type(pcd_o3d) == type(None):
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(points)
        pcd_o3d, normal_vectors = calculate_normals(pcd_o3d, k=30, radius=None)
        o3d.io.write_point_cloud("esf_in_pcd_temp.pcd", pcd_o3d)
        if np.array(points).shape[0] > 1000000:
            time.sleep(0.5)
    else:
        print("set one of these parameters: pcd_o3d --> Pointcloud object of o3d or points --> Array")

    # Run the .exe file and capture the output
    result = subprocess.run(
        [exe_path, "esf_in_pcd_temp.pcd"],  # Assuming executable prints result to stdout
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check = True
    )

    # Initialize variables
    esf_descriptor = []

    # Open and read the PCD file
    #try:
    with open("esf_in_pcd_temp_esf.pcd", 'r') as file:
        lines = file.readlines()
        data_start = False

        for line in lines:
            if line.startswith('DATA'):
                data_start = True
                continue

            if data_start:
                # Split the line into individual float values
                esf_descriptor = list(map(float, line.strip().split()))
    #except:
    #    print("ERROR")
    #    return esf_descriptor
    # Output the VFH descriptor
    #print(f"ESF Descriptor: {esf_descriptor}")
    #print(f"Number of elements: {len(esf_descriptor)}")

    # Check if the file exists
    # if os.path.exists("esf_in_pcd_temp.pcd"):
        #os.remove("esf_in_pcd_temp.pcd")

    if os.path.exists("esf_in_pcd_temp_esf.pcd"):
        os.remove("esf_in_pcd_temp_esf.pcd")

    return esf_descriptor

def preprocess_pcd_for_esf(pcd):
    # In NumPy-Array konvertieren
    points= np.asarray(pcd.points)
    # Zentrum berechnen und subtrahieren (Zentrierung)
    center = np.mean(points, axis=0)
    points_centered = points - center
    # Normieren auf [0, 1] (Skalierung durch max. Distanz)
    max_range = np.max(np.linalg.norm(points_centered, axis=1))
    points_scaled = points_centered / max_range
    pcd.points = o3d.utility.Vector3dVector(points_scaled)
    return pcd

def calculate_normals(pcd, k = 10, radius=None):
    if radius == None:
        radius = calculate_mean_radius(pcd, k)

    # Normalschätzung für die Punktwolke durchführen
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))

    # Normalenvektoren abrufen
    normal_vectors = np.round(np.asarray(pcd.normals),2)

    return pcd, normal_vectors

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
    mean_distances = np.mean(distances, axis=1)

    # Berechnung des Gesamtdurchschnitts
    overall_mean_distance = np.round(np.mean(mean_distances),3)

    # Ausgabe des Gesamtdurchschnitts
    #print("Overall mean distance:", overall_mean_distance)
    return overall_mean_distance, distances


def normalize_to_minus_one_and_one_v2(descriptor):
    max_abs_val = np.max(np.abs(descriptor))

    # Avoid division by zero if max_abs_val is zero
    if max_abs_val == 0:
        return np.zeros_like(descriptor)  # If all values are zero, return zero vector

    # Normalize to range [-1, 1] while keeping zero centered
    normalized_descriptor = descriptor / max_abs_val
    return normalized_descriptor
########################################################################################

##################################Normal HIST###########################################
# functions for generating normal hist features:
def generate_normal_xray_hist(pcd =None, points = None):
    if points == None:
        points = np.array(pcd.points)


    pcd_down_normal, downsampled_normal = voxel_downsample(points=points, bins=128)
    normal_hist = normals_distribution_histogram(downsampled_normal, bins_az=8, bins_el=8)

    # Reihenweise flatten:
    hist_flat = np.array(normal_hist).flatten(order='C')
    print("Normal Hist Shape:", np.array(normal_hist).shape)

    return list(normal_hist), list(hist_flat)

def voxel_downsample(pcd= None, points=None, bins= 64):
    if type(points) == type(None):
        points = np.array(pcd.points)
    voxel_size = 1.0 / bins
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    down = pcd.voxel_down_sample(voxel_size=voxel_size)
    return down, np.asarray(down.points)

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
##############################################################################################

##################################EXTRA FEATURES########################################
def calculate_cosine_similarity(esf_descriptor1, esf_descriptor2):
    """
    Berechnet die Kosinus-Ähnlichkeit zwischen den entsprechenden Teilhistogrammen
    von zwei ESF-Deskriptoren.

    Args:
    - esf_descriptor1: Erster ESF-Deskriptor (1x640 Vektor).
    - esf_descriptor2: Zweiter ESF-Deskriptor (1x640 Vektor).

    Returns:
    - Eine Liste mit 10 Werten, die die Kosinus-Ähnlichkeit für jedes der 10 Teilhistogramme darstellen.
    """
    # Anzahl der Bins pro Teilhistogramm
    bins_per_histogram = 64

    # Liste zum Speichern der Kosinus-Ähnlichkeiten für jedes Teilhistogramm
    cosine_similarities = []

    # Berechnung der Kosinus-Ähnlichkeit für jedes Teilhistogramm
    for i in range(10):
        start_index = i * bins_per_histogram
        end_index = start_index + bins_per_histogram

        # Entsprechende Teilhistogramme extrahieren
        histogram1 = esf_descriptor1[start_index:end_index]
        histogram2 = esf_descriptor2[start_index:end_index]

        # Kosinus-Ähnlichkeit berechnen
        if np.linalg.norm(normalize_to_minus_one_and_one_v2(histogram1)) == 0 and np.linalg.norm(normalize_to_minus_one_and_one_v2(histogram2)) != 0:
            similarity = 0  # oder eine andere logische Behandlung
        elif np.linalg.norm(normalize_to_minus_one_and_one_v2(histogram1)) != 0 and np.linalg.norm(normalize_to_minus_one_and_one_v2(histogram2)) ==0:
            similarity = 0  # oder eine andere logische Behandlung
        elif np.linalg.norm(normalize_to_minus_one_and_one_v2(histogram1)) == 0 and np.linalg.norm(normalize_to_minus_one_and_one_v2(histogram2)) == 0:
            similarity = 1
        else:
            similarity =  1 - cosine(normalize_to_minus_one_and_one_v2(histogram1), normalize_to_minus_one_and_one_v2(histogram2))
        cosine_similarities.append(similarity)
        #Cosine Similarity = 1 bedeutet, dass die beiden Vektoren in genau die gleiche Richtung zeigen, also maximal ähnlich sind.
        #Cosine Similarity = 0 bedeutet, dass die Vektoren orthogonal zueinander sind, also keine Ähnlichkeit haben.
        #Cosine Similarity = -1 bedeutet, dass die Vektoren in entgegengesetzte Richtungen zeigen, also maximal unähnlich sind.
    return cosine_similarities # list with values in range [-1,1]


def compute_emd_for_esf(descriptor1, descriptor2):
    """
    Berechnet den Earth Mover's Distance (EMD) zwischen zwei ESF-Deskriptoren für jedes Teilhistogramm.

    Parameters:
    - descriptor1: Erster ESF-Deskriptor (640-dimensionaler Vektor)
    - descriptor2: Zweiter ESF-Deskriptor (640-dimensionaler Vektor)

    Returns:
    - emd_values: Liste von 10 Werten, die den EMD für jedes Teilhistogramm repräsentieren
    """

    # Überprüfen, ob beide Deskriptoren die richtige Länge haben
    if len(descriptor1) != 640 or len(descriptor2) != 640:
        raise ValueError("Beide Deskriptoren müssen eine Länge von 640 haben.")

    emd_values = []

    # Berechnen des EMD für jedes Teilhistogramm (64 Bins pro Histogramm)
    for i in range(10):
        start_index = i * 64
        end_index = start_index + 64

        # Extrahiere das Teilhistogramm
        histogram1 = descriptor1[start_index:end_index]
        histogram2 = descriptor2[start_index:end_index]

        # Berechne den EMD (Wasserstein-Distance) zwischen den beiden Teilhistogrammen
        emd = wasserstein_distance(normalize_to_minus_one_and_one_v2(histogram1), normalize_to_minus_one_and_one_v2(histogram2))

        # Füge den EMD-Wert zur Liste hinzu
        emd_values.append(emd)

    return emd_values
########################################################################################################################

#############################GRID#####################################
# functions for generating grid features:

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

def _symmetric_capped_chamfer(
    A: np.ndarray, B: np.ndarray, cap_radius: float, sample_max: int
) -> float:
    """Schnelle sym. Chamfer mit Cap. Subsamplet große Clouds."""
    A = np.asarray(A, dtype=np.float32).reshape(-1, 3)
    B = np.asarray(B, dtype=np.float32).reshape(-1, 3)
    if A.size==0 and B.size==0:
        return 0.0
    if A.size==0:
        return float(cap_radius)
    if B.size==0:
        return float(cap_radius)

    if len(A) > sample_max:
        idx = np.random.choice(len(A), size=sample_max, replace=False)
        A = A[idx]
    if len(B) > sample_max:
        idx = np.random.choice(len(B), size=sample_max, replace=False)
        B = B[idx]

    tA = cKDTree(A); tB = cKDTree(B)
    da, _ = tB.query(A, k=1)
    db, _ = tA.query(B, k=1)
    da = np.clip(da, 0.0, cap_radius)
    db = np.clip(db, 0.0, cap_radius)
    return float(0.5*(da.mean() + db.mean()))

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


# used Dataset Class format:
class ESFRefPairDatasetMLP704Fast(Dataset):
    """
    Liefert:
      x704 : abs(esf_ref - esf_scan) [640]  ||  aug(abs(norm_ref - norm_scan)) [64]  -> (704,)
      xext : optionale 44d-Extras (memmap/json oder on-the-fly)
      y    : 0 (anderes Objekt; ehem. label=2) / 1 (vorhanden)
    label==0 wird vollständig verworfen.
    """

    # ----------------------------- Init ----------------------------- #
    def __init__(self,
                 esf_path, normal_path, cv_path, fold: str,
                 use_metrics=False, extra_feats_path=None, grid_feats_path=None,
                 max_negatives_train=500_000, max_negatives_val=None, max_negatives_test=None,
                 rng_seed=42, aug_rotate_norm=True, aug_noise_std=0.01,
                 drop_pos_perc_values=None,          # z.B. {"100|30","100|35","100|40","100|45"}
                 drop_pos_where = "either", # "ref" | "scan" | "either"
                 debug = True, xext_policy="all"  # <<< NEU: "as_is" | "extra" | "grid" | "all"
                 ):
        self.xext_policy = str(xext_policy).lower()
        self.use_metrics     = bool(use_metrics)
        self._feat_mode      = None     # "memmap" | "json" | None
        self.rng             = np.random.default_rng(rng_seed)
        self.aug_rotate_norm = bool(aug_rotate_norm)
        self.aug_noise_std   = float(aug_noise_std)
        self.drop_pos_perc_values = set(drop_pos_perc_values or [])
        self.drop_pos_where = str(drop_pos_where).lower()
        self.debug = bool(debug)
        self.drop_pos_perc_values = set(drop_pos_perc_values or [])
        self._ban_perc = {self._canon_perc(x) for x in self.drop_pos_perc_values}
        if self.drop_pos_where not in {"ref", "scan", "either"}:
            raise ValueError("drop_pos_where must be one of {'ref','scan','either'}")

        # Rohdaten laden
        with open(esf_path,   "rb") as f: self.esf  = orjson.loads(f.read())
        with open(normal_path,"rb") as f: self.norm = orjson.loads(f.read())

        # CV laden
        with open(cv_path, "r", encoding="utf-8") as f:
            cv = json.load(f)[fold]
        self.train_raw, self.val_raw, self.test_raw = cv["train"], cv["val"], cv["test"]

        # Extra-Features (memmap/json) optional
        self._load_precomputed_feats(extra_feats_path, grid_feats_path, fold)

        # 1) Paare bauen + _orig_idx setzen
        self._build_indices()

        # 2) Vorher-Counts
        self._report_split_counts("BEFORE")

        # 3) Negatives (label==2) pro Split kappen
        self._apply_negative_cap(max_negatives_train, max_negatives_val, max_negatives_test)

        # 4) Stores auf benötigte Keys schrumpfen
        self._shrink_hist_stores()

        # 5) Extra-Features schrumpfen (robust; Memmap: OOB filtern)
        self._shrink_extra_features()

        # 6) Indizes nach Shrink neu
        self._rebuild_indices_after_shrink()

        # 7) Nachher-Counts
        self._report_split_counts("AFTER")

        # Konsistenz (robuster Modus: nur prüfen, ob Mappings existieren)
        if self._feat_mode == "memmap":
            assert hasattr(self, "_feat_tr_idx") and hasattr(self, "_feat_va_idx") and hasattr(self, "_feat_te_idx")
        elif getattr(self, "pre_feats", None) is not None:
            expected = len(self.train_pairs) + len(self.val_pairs) + len(self.test_pairs)
            assert len(self.pre_feats) == expected, f"pre_feats {len(self.pre_feats)} != erwartet {expected}"

        print(f"✓ Dataset ready | Train {len(self.train_pairs)} | Val {len(self.val_pairs)} | Test {len(self.test_pairs)}")

    # ----------------------------- Helpers ----------------------------- #
    def _y_mapped(self, p):
        # 0 => verwerfen; 2 => 0; 1 => 1
        y = p["label"]
        if y == 0:
            return None
        return 0 if y == 2 else 1

    @staticmethod
    def _parse_key(key: str):
        p = key.split("_")
        return "_".join(p[:-3]), p[-3], p[-2], int(p[-1])

    @staticmethod
    def _safe_get(store, cls, inst, perc, idx, exp_len):
        try:
            v = store[cls][str(inst)][str(perc)][idx]
            a = np.asarray(v, np.float32)
            return a if a.size == exp_len else None
        except Exception:
            return None

    @staticmethod
    def _row_metrics(v1, v2):
        cos_local = 1 - _cos(v1, v2)
        v1n = v1 / (np.linalg.norm(v1) + 1e-8)
        v2n = v2 / (np.linalg.norm(v2) + 1e-8)
        cos_global = 1 - _cos(v1n, v2n)
        emd_local  = wasserstein_distance(v1, v2) / 64.0
        emd_global = wasserstein_distance(v1n, v2n) / 64.0
        # robust
        if not np.isfinite(cos_local):  cos_local = 0.0
        if not np.isfinite(cos_global): cos_global = 0.0
        if not np.isfinite(emd_local):  emd_local  = 1.0
        if not np.isfinite(emd_global): emd_global = 1.0
        return cos_local, cos_global, emd_local, emd_global

    def _safe_load_grid(self, path):
        try:
            # bevorzugt: echtes float-Array, memmapfähig
            arr = np.load(path, mmap_mode="r")
            return arr, True  # memmap_ok
        except ValueError as e:
            if "allow_pickle=False" in str(e):
                # Fallback: Pickle zulassen, aber OHNE memmap
                arr = np.load(path, allow_pickle=True)
                # falls object-Array: stapeln + casten
                if arr.dtype == object:
                    arr = np.stack(arr, axis=0).astype(np.float32, copy=False)
                return arr, False
            raise

    def _load_precomputed_feats(self, path, grid_path, fold):
        self.pre_feats = None
        self._feat_mode = None
        # GRID defaults
        self._grid_mode = None
        self._gfeat_tr = self._gfeat_va = self._gfeat_te = None
        self._gfeat_tr_idx = self._gfeat_va_idx = self._gfeat_te_idx = None
        self._grid_dim = 27  # wird unten ggf. aus Datei gelesen
        # --- 44er Extras ---
        if path is not None:
            if os.path.isdir(path):
                tr_p = os.path.join(path, f"{fold}_train.npy")
                va_p = os.path.join(path, f"{fold}_val.npy")
                te_p = os.path.join(path, f"{fold}_test.npy")
                for pth in (tr_p, va_p, te_p):
                    if not os.path.exists(pth):
                        raise FileNotFoundError(f"Fehlt: {pth}")
                self._feat_tr = np.load(tr_p, mmap_mode="r")
                self._feat_va = np.load(va_p, mmap_mode="r")
                self._feat_te = np.load(te_p, mmap_mode="r")
                self._feat_tr_idx = self._feat_va_idx = self._feat_te_idx = None
                self._feat_mode = "memmap"
            else:
                self._feat_mode = "json"
                if str(path).lower().endswith(".gz"):
                    import gzip
                    with gzip.open(path, "rb") as f:
                        data = orjson.loads(f.read())
                else:
                    with open(path, "rb") as f:
                        data = orjson.loads(f.read())
                fld = data.get(fold, {})
                self.pre_feats = fld.get("train", []) + fld.get("val", []) + fld.get("test", [])

        # --- 27er GRID ---
        if grid_path is not None:
            if not os.path.isdir(grid_path):
                raise ValueError(f"grid_feats_path muss ein Ordner sein: {grid_path}")
            gtr = os.path.join(grid_path, f"{fold}_train.npy")
            gva = os.path.join(grid_path, f"{fold}_val.npy")
            gte = os.path.join(grid_path, f"{fold}_test.npy")
            for pth in (gtr, gva, gte):
                if not os.path.exists(pth):
                    raise FileNotFoundError(f"Grid-Features fehlen: {pth}")

            self._gfeat_tr, tr_memmap = self._safe_load_grid(gtr)
            self._gfeat_va, va_memmap = self._safe_load_grid(gva)
            self._gfeat_te, te_memmap = self._safe_load_grid(gte)

            if self._gfeat_tr.ndim == 2:
                self._grid_dim = int(self._gfeat_tr.shape[1])
            self._grid_mode = "memmap"

    # ----------------------------- Build indices ----------------------------- #
    # --- Neu: Hilfsfunktion ---
    @staticmethod
    def _canon_perc(v):
        s = str(v).strip()
        # nimm den rechten Teil, falls "100|30" etc.
        if "|" in s:
            s = s.split("|")[-1].strip()
        return s

    def _is_banned_positive(self, p):
        # nur positive Paare (label==1) prüfen
        if p.get("label") != 1 or not self._ban_perc:
            return False

        cls_r, inst_r, perc_r, idx_r = self._parse_key(p["esf_ref"])
        cls_s, inst_s, perc_s, idx_s = self._parse_key(p["esf_scan"])

        # kanonisieren (z.B. "100|30" -> "30")
        pr = self._canon_perc(perc_r)
        ps = self._canon_perc(perc_s)

        ref_bad = (self.drop_pos_where in {"ref", "either"}) and (pr in self._ban_perc)
        scan_bad = (self.drop_pos_where in {"scan", "either"}) and (ps in self._ban_perc)
        banned = ref_bad or scan_bad

        if banned and self.debug:
            who = []
            if ref_bad:  who.append("ref")
            if scan_bad: who.append("scan")
            print(f"[DROP POS] perc_ref={perc_r}({pr}) perc_scan={perc_s}({ps}) -> via {','.join(who)}")

        return banned

    def _keep_pair(self, p):
        if p["label"] == 0:
            return False
        # <<< NEU: positives per 'perc' filtern
        if self._is_banned_positive(p):
            return False

        cls_r, inst_r, perc_r, idx_r = self._parse_key(p["esf_ref"])
        cls_s, inst_s, perc_s, idx_s = self._parse_key(p["esf_scan"])
        e_r = self._safe_get(self.esf, cls_r, inst_r, perc_r, idx_r, 640)
        e_s = self._safe_get(self.esf, cls_s, inst_s, perc_s, idx_s, 640)
        n_r = self._safe_get(self.norm, cls_r, inst_r, perc_r, idx_r, 64)
        n_s = self._safe_get(self.norm, cls_s, inst_s, perc_s, idx_s, 64)
        return (e_r is not None) and (e_s is not None) and (n_r is not None) and (n_s is not None)


    def _build_indices(self):
        self.train_pairs = []
        for i, p in enumerate(self.train_raw):
            if self._keep_pair(p):
                q = dict(p); q["_orig_idx"] = i
                self.train_pairs.append(q)

        self.val_pairs = []
        for i, p in enumerate(self.val_raw):
            if self._keep_pair(p):
                q = dict(p); q["_orig_idx"] = i
                self.val_pairs.append(q)

        self.test_pairs = []
        for i, p in enumerate(self.test_raw):
            if self._keep_pair(p):
                q = dict(p); q["_orig_idx"] = i
                self.test_pairs.append(q)

        self.all_pairs  = self.train_pairs + self.val_pairs + self.test_pairs
        # Offsets im originalen pre_feats-Layout (train + val + test)
        self._train_off = (0, len(self.train_pairs))
        self._val_off   = (len(self.train_pairs), len(self.train_pairs) + len(self.val_pairs))
        self._test_off  = (len(self.train_pairs) + len(self.val_pairs), len(self.all_pairs))
        self._rebuild_indices_after_shrink()

    def _rebuild_indices_after_shrink(self):
        self.all_pairs = self.train_pairs + self.val_pairs + self.test_pairs
        self.train_idx = np.arange(0, len(self.train_pairs))
        self.val_idx   = np.arange(len(self.train_pairs), len(self.train_pairs) + len(self.val_pairs))
        self.test_idx  = np.arange(len(self.all_pairs) - len(self.test_pairs), len(self.all_pairs))

    # ----------------------------- Reporting ----------------------------- #
    def _report_split_counts(self, tag):
        def counts(pairs):
            n1 = sum(1 for p in pairs if p["label"] == 1)
            n2 = sum(1 for p in pairs if p["label"] == 2)
            n0 = sum(1 for p in pairs if p["label"] == 0)
            return len(pairs), n1, n2, n0

        t_all, t1, t2, t0 = counts(self.train_pairs)
        v_all, v1, v2, v0 = counts(self.val_pairs)
        s_all, s1, s2, s0 = counts(self.test_pairs)

        print(f"[{tag}] COUNTS")
        print(f"  Train: total={t_all} | label1={t1} | label2={t2} | label0={t0}")
        print(f"  Val  : total={v_all} | label1={v1} | label2={v2} | label0={v0}")
        print(f"  Test : total={s_all} | label1={s1} | label2={s2} | label0={s0}")

    # ----------------------------- Shrinks ----------------------------- #
    def _apply_negative_cap(self, max_negatives_train=500_000, max_negatives_val=None, max_negatives_test=None):
        def cap_split(pairs, cap):
            pos = [p for p in pairs if p["label"] == 1]
            neg = [p for p in pairs if p["label"] == 2]
            if cap is not None and len(neg) > cap:
                sel = set(self.rng.choice(len(neg), size=cap, replace=False).tolist())
                neg = [neg[i] for i in sel]
            out = pos + neg
            self.rng.shuffle(out)
            return out

        self.train_pairs = cap_split(self.train_pairs, max_negatives_train)
        self.val_pairs   = cap_split(self.val_pairs,   max_negatives_val)
        self.test_pairs  = cap_split(self.test_pairs,  max_negatives_test)

    def _needed_hist_keys(self):
        need = set()
        for pairs in (self.train_pairs, self.val_pairs, self.test_pairs):
            for p in pairs:
                for k in ("esf_ref", "esf_scan"):
                    cls, inst, perc, idx = self._parse_key(p[k])
                    need.add(("esf",  cls, str(inst), str(perc), idx))
                    need.add(("norm", cls, str(inst), str(perc), idx))
        return need

    def _shrink_hist_stores(self):
        need = self._needed_hist_keys()

        def shrink_one(tag_name, store):
            out = {}
            for tag, cls, inst, perc, idx in need:
                if tag != tag_name: continue
                cls_d  = out.setdefault(cls, {})
                inst_d = cls_d.setdefault(inst, {})
                perc_d = inst_d.setdefault(perc, {})
                try:
                    val = store[cls][inst][perc][idx]
                except Exception:
                    continue
                perc_d[idx] = val
            return out

        self.esf  = shrink_one("esf",  self.esf)
        self.norm = shrink_one("norm", self.norm)
        gc.collect()

    def _shrink_extra_features(self):
        import gc
        if self._feat_mode == "json":
            # unverändert wie bisher ...
            t0, t1 = self._train_off;
            v0, v1 = self._val_off;
            s0, s1 = self._test_off
            pre_tr = self.pre_feats[t0:t1];
            pre_va = self.pre_feats[v0:v1];
            pre_te = self.pre_feats[s0:s1]

            def block_new_feats(pairs, block_feats):
                return [block_feats[p["_orig_idx"]] for p in pairs]

            new_feats = []
            new_feats.extend(block_new_feats(self.train_pairs, pre_tr))
            new_feats.extend(block_new_feats(self.val_pairs, pre_va))
            new_feats.extend(block_new_feats(self.test_pairs, pre_te))
            self.pre_feats = new_feats
            gc.collect()

        elif self._feat_mode == "memmap":
            # 44er-Extras: nur Indexmaps bauen
            self._feat_tr_idx = np.array([p["_orig_idx"] for p in self.train_pairs], dtype=np.int64)
            self._feat_va_idx = np.array([p["_orig_idx"] for p in self.val_pairs], dtype=np.int64)
            self._feat_te_idx = np.array([p["_orig_idx"] for p in self.test_pairs], dtype=np.int64)
            # OOB lassen wir zu; _get_extra gibt dann leeren Tensor/Nullen zurück

        # GRID-Memmaps: ebenfalls Indexmaps (falls vorhanden)
        if self._grid_mode == "memmap":
            self._gfeat_tr_idx = np.array([p["_orig_idx"] for p in self.train_pairs], dtype=np.int64)
            self._gfeat_va_idx = np.array([p["_orig_idx"] for p in self.val_pairs], dtype=np.int64)
            self._gfeat_te_idx = np.array([p["_orig_idx"] for p in self.test_pairs], dtype=np.int64)

    # ----------------------------- Core features ----------------------------- #
    def _diff704(self, e_r, e_s, n_r, n_s):
        de = np.abs(e_r - e_s).astype(np.float32)  # (640,)
        dn = np.abs(n_r - n_s).astype(np.float32)  # (64,)

        # Rotation nur auf NormalHist (8x8) – optional
        if self.aug_rotate_norm and (self.rng.random() < 0.5):
            dn = np.roll(dn.reshape(8, 8), self.rng.integers(1, 8), axis=0).reshape(64)

        x = np.concatenate([de, dn], 0)            # (704,)

        # Noise auf ALLE 704 – optional
        if (self.aug_noise_std > 0.0) and (self.rng.random() < 0.5):
            x = x + self.rng.normal(0.0, self.aug_noise_std, size=x.shape).astype(np.float32)

        return x

    def _get_extra(self, i):
        import torch, numpy as np

        # ------- 44D laden (wie bei dir, unverändert bis ext44/grid27 gebaut sind) -------
        # ext44
        ext44 = None
        if self.use_metrics:
            ext44 = np.empty((0,), dtype=np.float32)  # oder deine on-the-fly-Features
        elif self._feat_mode is None:
            ext44 = np.empty((0,), dtype=np.float32)
        elif self._feat_mode == "json":
            ext44 = np.asarray(self.pre_feats[i], dtype=np.float32)
        elif self._feat_mode == "memmap":
            n_tr = len(self.train_pairs);
            n_va = len(self.val_pairs)
            if i < n_tr:
                j = int(self._feat_tr_idx[i]);
                src = self._feat_tr
            elif i < n_tr + n_va:
                j = int(self._feat_va_idx[i - n_tr]);
                src = self._feat_va
            else:
                j = int(self._feat_te_idx[i - n_tr - n_va]);
                src = self._feat_te
            if 0 <= j < len(src):
                ext44 = np.asarray(src[j], dtype=np.float32)
            else:
                ext44 = np.empty((0,), dtype=np.float32)

        # grid27
        grid27 = None
        if self._grid_mode == "memmap":
            n_tr = len(self.train_pairs);
            n_va = len(self.val_pairs)
            if i < n_tr:
                j = int(self._gfeat_tr_idx[i]);
                src = self._gfeat_tr
            elif i < n_tr + n_va:
                j = int(self._gfeat_va_idx[i - n_tr]);
                src = self._gfeat_va
            else:
                j = int(self._gfeat_te_idx[i - n_tr - n_va]);
                src = self._gfeat_te
            if 0 <= j < len(src):
                grid27 = np.asarray(src[j], dtype=np.float32)
            else:
                grid27 = np.zeros((self._grid_dim,), dtype=np.float32)
        else:
            grid27 = np.empty((0,), dtype=np.float32)

        # ------- Fix-Längen Helpers -------
        def _fix_len(x: np.ndarray, D: int) -> np.ndarray:
            if x is None or x.size == 0:
                return np.zeros((D,), dtype=np.float32)
            x = x.reshape(-1).astype(np.float32, copy=False)
            if x.size == D:
                return x
            if x.size > D:
                return x[:D]
            out = np.zeros((D,), dtype=np.float32)
            out[:x.size] = x
            return out

        Dg = int(self._grid_dim)  # meist 27
        pol = getattr(self, "xext_policy", "as_is").lower()

        if pol == "extra":
            # immer 44D
            return torch.from_numpy(_fix_len(ext44, 44))

        if pol == "grid":
            # immer 27D
            return torch.from_numpy(_fix_len(grid27, Dg))

        if pol == "all":
            # immer 71D (44 + Dg)
            x44 = _fix_len(ext44, 44)
            x27 = _fix_len(grid27, Dg)
            return torch.from_numpy(np.concatenate([x44, x27], 0))

        # Fallback: altes Verhalten (kann mischen – NICHT empfohlen)
        if (ext44 is None or ext44.size == 0) and (grid27 is None or grid27.size == 0):
            return torch.tensor([])
        xcat = np.concatenate([ext44, grid27], 0).astype(np.float32, copy=False)
        return torch.from_numpy(xcat)
    # ----------------------------- Dataset API ----------------------------- #
    def __len__(self): return len(self.all_pairs)

    def __getitem__(self, idx):
        start = idx
        while True:
            p = self.all_pairs[idx]
            y = 0 if p["label"] == 2 else 1

            cls_r, inst_r, perc_r, idx_r = self._parse_key(p["esf_ref"])
            cls_s, inst_s, perc_s, idx_s = self._parse_key(p["esf_scan"])
            e_r = self._safe_get(self.esf, cls_r, inst_r, perc_r, idx_r, 640)
            e_s = self._safe_get(self.esf, cls_s, inst_s, perc_s, idx_s, 640)
            n_r = self._safe_get(self.norm, cls_r, inst_r, perc_r, idx_r, 64)
            n_s = self._safe_get(self.norm, cls_s, inst_s, perc_s, idx_s, 64)

            if (e_r is None) or (e_s is None) or (n_r is None) or (n_s is None):
                idx = (idx + 1) % len(self.all_pairs)
                if idx == start:
                    raise RuntimeError("Kein gültiges Sample im Dataset.")
                continue

            x704 = self._diff704(e_r, e_s, n_r, n_s)
            xext = self._get_extra(idx)
            return torch.from_numpy(x704), xext, torch.tensor(y, dtype=torch.long)#, torch.tensor(idx, dtype=torch.long)

    # ----------------------------- Loaders ----------------------------- #
    def get_loaders(self, batch_size=128, batch_size_val=None, num_workers=0,
                    use_weighted_sampler=False):
        val_bs = batch_size_val if batch_size_val is not None else batch_size

        ds_train = Subset(self, self.train_idx)
        ds_val   = Subset(self, self.val_idx)
        ds_test  = Subset(self, self.test_idx)

        pw = bool(num_workers > 0)
        if use_weighted_sampler:
            # inverse Häufigkeit auf train_idx (nach Kappung/Shuffle)
            n0 = sum(1 for i in self.train_idx if (0 if self.all_pairs[i]["label"] == 2 else 1) == 0)
            n1 = len(self.train_idx) - n0
            w0 = 1.0 / max(n0, 1)
            w1 = 1.0 / max(n1, 1)
            ws = [(w0 if (0 if self.all_pairs[i]["label"] == 2 else 1) == 0 else w1) for i in self.train_idx]
            sampler = WeightedRandomSampler(ws, num_samples=len(ws), replacement=True)
            train_loader = DataLoader(ds_train, batch_size=batch_size, sampler=sampler,
                                      pin_memory=True, num_workers=num_workers,
                                      persistent_workers=pw, drop_last=False)
        else:
            train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                                      pin_memory=True, num_workers=num_workers,
                                      persistent_workers=pw, drop_last=False)

        val_loader  = DataLoader(ds_val,  batch_size=val_bs, shuffle=False,
                                 pin_memory=True, num_workers=num_workers,
                                 persistent_workers=pw, drop_last=False)
        test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False,
                                 pin_memory=True, num_workers=num_workers,
                                 persistent_workers=pw, drop_last=False)
        return train_loader, val_loader, test_loader



def preprocess_data_for_mlp_input(file_path_target, file_path_source):
    model_target_pcd = o3d.io.read_point_cloud(file_path_target)
    scan_source_pcd = o3d.io.read_point_cloud(file_path_source)

    # preprocessing for esf and xray normal hist vector genneration
    model_target_pcd_esf  = preprocess_pcd_for_esf(pcd_model_target)
    scan_source_pcd_esf = preprocess_pcd_for_esf(pcd_scan_source)

    #esf features generation
    esf_planned = compute_esf_descriptor(pcd_o3d=pcd_model_target_esf)
    esf_scan = compute_esf_descriptor(pcd_o3d=pcd_scan_source_esf)

    # esf differences
    esf_diff = np.array(esf_planned) - np.array(esf_scan)
    # normalization of esf differences
    esf_scan_diff_norm = normalize_to_minus_one_and_one_v2(esf_diff)

    # calculating extra feats (emd and cosine)
    emd_values =  compute_emd_for_esf(esf_planned, esf_scan)
    cosine_similarity_values = calculate_cosine_similarity(esf_planned, esf_scan)

    esf_diff_extended_norm  = np.concatenate((esf_scan_diff_norm, emd_values, cosine_similarity_values))
    esf_diff_extended_norm  =np.round(esf_diff_extended_norm, decimals=4)


    # normal hist

    points = np.asarray(pcd.points)
    model_target_xray_histo_vector = generate_normal_xray_hist(pcd_model_target_esf)[1]
    scan_source_xray_histo_vector = generate_normal_xray_hist(pcd_scan_source_esf)[1]

    # grid features
    grid_vector = features_for_object(pcd_ref_o3d=pcd, pcd_scan_obj_o3d=pcd, voxel_kwargs=voxel_kwargs)

    return
