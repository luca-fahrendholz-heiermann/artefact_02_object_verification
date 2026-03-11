import time

import numpy as np
import math
import random
import open3d as o3d
import subprocess
import os
import copy
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from scipy.spatial import ConvexHull

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

def calculate_mean_radius_exe(pcd, k = 10):

    # Berechnen der Entfernungen und Indizes der k-nächsten Nachbarn für jeden Punkt
    distances =  compute_distances_kdtree_k_search(pcd, k)

    # Berechnung des durchschnittlichen Abstands für jeden Punkt
    mean_distances = np.mean(distances, axis=1)

    # Berechnung des Gesamtdurchschnitts
    overall_mean_distance = np.round(np.mean(mean_distances),3)

    # Ausgabe des Gesamtdurchschnitts
    print("Overall mean distance:", overall_mean_distance)
    return overall_mean_distance

def calculate_normals(pcd, k = 10, radius=None):
    if radius == None:
        radius = calculate_mean_radius(pcd, k)

    # Normalschätzung für die Punktwolke durchführen
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))

    # Normalenvektoren abrufen
    normal_vectors = np.round(np.asarray(pcd.normals),2)

    return pcd, normal_vectors

# VFH Descriptor
def compute_vfh_vector(pcd_o3d=None, points = None, exe_path = os.path.join(os.getcwd(), "pcl_vfh_estimation_debug.exe")):
    if type(points) == type(None):
        # Save the point cloud as a PCD file
        o3d.io.write_point_cloud("vfh_in_pcd_temp.pcd", pcd_o3d)
    elif type(pcd_o3d) == type(None):
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud("vfh_in_pcd_temp.pcd", pcd_o3d)
    else:
        print("set one of these parameters: pcd_o3d --> Pointcloud object of o3d or points --> Array")

    # Run the .exe file and capture the output
    result = subprocess.run([exe_path, "vfh_in_pcd_temp.pcd", "vfh_out_temp.pcd"],  # Assuming executable prints result to stdout
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Initialize variables
    vfh_descriptor = []

    # Open and read the PCD file
    with open("vfh_out_temp.pcd", 'r') as file:
        lines = file.readlines()
        data_start = False

        for line in lines:
            if line.startswith('DATA'):
                data_start = True
                continue

            if data_start:
                # Split the line into individual float values
                vfh_descriptor = list(map(float, line.strip().split()))

    # Output the VFH descriptor
    print(f"VFH Descriptor: {vfh_descriptor}")
    print(f"Number of elements: {len(vfh_descriptor)}")

    # Check if the file exists
    if os.path.exists("esf_out_temp.pcd"):
        os.remove("esf_out_temp.pcd")

    if os.path.exists("esf_in_pcd_temp.pcd"):
        os.remove("esf_in_pcd_temp.pcd")

    return vfh_descriptor

def compute_harris3d_keypoints(pcd_o3d=None, points = None, exe_path = os.path.join(os.getcwd(), "harris3d_estimation.exe")):
    if type(points) == type(None):
        # Save the point cloud as a PCD file
        o3d.io.write_point_cloud("harris_in_pcd_temp.pcd", pcd_o3d)
    elif type(pcd_o3d) == type(None):
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud("harris_in_pcd_temp.pcd", pcd_o3d)
    else:
        print("set one of these parameters: pcd_o3d --> Pointcloud object of o3d or points --> Array")

    # Run the .exe file and capture the output
    result = subprocess.run(
        [exe_path, "harris_in_pcd_temp.pcd"],  # Assuming executable prints result to stdout
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    pcd_kp = o3d.io.read_point_cloud(os.path.join(os.getcwd(),"keypoints.pcd"))
    o3d.visualization.draw_geometries([pcd_kp.paint_uniform_color([1,0,0]), pcd_o3d.paint_uniform_color([0.5,0.5,0.5])])
    o3d.visualization.draw_geometries([pcd_kp])
    # Initialize variables
    keypoints = (pcd_kp.points)


    # Output the VFH descriptor
    #print(f"ESF Descriptor: {esf_descriptor}")
    #print(f"Number of elements: {len(esf_descriptor)}")

    # Check if the file exists
    if os.path.exists("esf_in_pcd_temp.pcd"):
        os.remove("harris_in_pcd_temp.pcd")

    if os.path.exists("esf_in_pcd_temp_esf.pcd"):
        os.remove("keypoints.pcd")

    return pcd_kp

def plot_histogram(descriptor, label, ylim = None ):
    plt.figure(figsize=(10, 5))

    # Each bin corresponds to one value in the 640-dimensional vector
    plt.bar(range(len(descriptor)), descriptor, width=1.0, edgecolor='black')

    plt.xlabel('Vector Index (0 to 639)')
    plt.ylabel('Normalized Value (-1 to 1)')
    plt.title('Histogram of ESF Descriptor Differences - ' + label)
    if ylim == None:
        #plt.ylim(-1, 1)  # Set the y-axis limits to -1 to 1
        plt.xlim(0, len(descriptor))  # Set the x-axis limits from 0 to 640
    else:
        plt.ylim(ylim[0], ylim[1])  # Set the y-axis limits to -1 to 1

    plt.show()

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

# # Output the VFH descriptor
# print(f"VFH Descriptor: {vfh_descriptor}")
# print(f"Number of elements: {len(vfh_descriptor)}")

def compute_distances_kdtree_k_search(pcd_o3d=None, k = 10, points = None, exe_path = os.path.join(os.getcwd(), "kdtree_search_k.exe")):
    if type(points) == type(None):
        # Save the point cloud as a PCD file
        o3d.io.write_point_cloud("kdtree_pcd.pcd", pcd_o3d)
    elif type(pcd_o3d) == type(None):
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud("kdtree_pcd.pcd", pcd_o3d)
    else:
        print("set one of these parameters: pcd_o3d --> Pointcloud object of o3d or points --> Array")

    # Run the .exe file and capture the output
    result = subprocess.run([exe_path, os.path.join(os.getcwd(), "kdtree_pcd.pcd"), "-k", str(k)],
                            # Assuming executable prints result to stdout
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            check = True
                            )

    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    else:
        print(f"Output: {result.stdout}")

    distances = np.loadtxt('distances.txt')

    print(np.array(pcd_o3d.points).shape)
    print(distances.shape)
    print(distances)

    if os.path.exists(os.path.join(os.getcwd(), "kdtree_pcd.pcd")):
        os.remove(os.path.join(os.getcwd(), "kdtree_pcd.pcd"))

    if os.path.exists(os.path.join(os.getcwd(), "distances.txt")):
        os.remove(os.path.join(os.getcwd(), "distances.txt"))

        #distances.txt
    return distances


def compute_distances_kdtree_r_search(pcd_o3d=None, r = 1.0, points = None, exe_path = os.path.join(os.getcwd(), "kdtree_search_radius.exe")):
    if type(points) == type(None):
        # Save the point cloud as a PCD file
        o3d.io.write_point_cloud("kdtree_pcd.pcd", pcd_o3d)
    elif type(pcd_o3d) == type(None):
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud("kdtree_pcd.pcd", pcd_o3d)
    else:
        print("set one of these parameters: pcd_o3d --> Pointcloud object of o3d or points --> Array")

    # # Run the .exe file and capture the output
    # result = subprocess.run([exe_path, os.path.join(os.getcwd(), "kdtree_pcd.pcd"), "-r", str(r)],
    #                         # Assuming executable prints result to stdout
    #                         stdout=subprocess.PIPE,
    #                         stderr=subprocess.PIPE,
    #                         text=True,
    #                         check = True
    #                         )
    #
    # if result.returncode != 0:
    #     print(f"Error: {result.stderr}")
    # else:
    #     print(f"Output: {result.stdout}")
    #
    # neighbours = np.loadtxt('neighbours.txt')

    # Run the C++ executable and capture the output
    result = subprocess.run([exe_path, os.path.join(os.getcwd(), "kdtree_pcd.pcd"), "-r", str(r)],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True)

    if result.returncode != 0:
        print("Error:", result.stderr)
        return None

    # Split the output by lines and convert to a numpy array
    neighbors = np.array(result.stdout.strip().split('\n'), dtype=int)

    print(np.array(pcd_o3d.points).shape)
    print(neighbors.shape)
    print(neighbors)

    if os.path.exists(os.path.join(os.getcwd(), "kdtree_pcd.pcd")):
        os.remove(os.path.join(os.getcwd(), "kdtree_pcd.pcd"))

    if os.path.exists(os.path.join(os.getcwd(), "neighbours.txt")):
        os.remove(os.path.join(os.getcwd(), "neighbours.txt"))

        #distances.txt
    return neighbors
def calculate_density(pcd, k = 10, radius = None):
    if radius == None:
        radius = calculate_mean_radius(pcd, k)


    point_cloud = np.array(pcd.points)
    # Build KDTree for efficient nearest neighbor search
    kdtree = KDTree(point_cloud)

    # Query neighbors within the specified radius for each point
    num_neighbors = kdtree.query_radius(point_cloud, r=radius, count_only=True)

    # Normalize density by total number of points
    density = num_neighbors

    return density

def calculate_density_exe(pcd, k = 10, radius = None):
    if radius == None:
        radius = calculate_mean_radius(pcd, k)

    # Query neighbors within the specified radius for each point
    num_neighbors = compute_distances_kdtree_r_search(pcd, radius)

    # Normalize density by total number of points
    density = num_neighbors

    return density


def calculate_density_histogram(pcd, k=10, radius=None, bins=64, points = None):
    if points == None:
        pcd = pcd
    else:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

    if radius is None:
        start = time.time()
        radius, distances = calculate_mean_radius(pcd, k)
        print("radius calc: ", time.time()-start)

    point_cloud = np.array(pcd.points)

    # KDTree für effiziente Nächste-Nachbarn-Suche erstellen
    start = time.time()
    kdtree = KDTree(point_cloud)

    # Nachbarn innerhalb des spezifizierten Radius für jeden Punkt abfragen
    num_neighbors = kdtree.query_radius(point_cloud, r=radius, count_only=True)
    print("knn: ", time.time()-start)

    # Dichte wird durch die Anzahl der Nachbarn in diesem Radius repräsentiert
    density = num_neighbors

    mean_density = np.mean(normalize_to_minus_one_and_one_v2(density))

    # Erstellen eines Histogramms mit 64 Bins
    histogram, bin_edges = np.histogram(density, bins=bins, density=True)

    ## Mittelpunkte der Bins berechnen
    #bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Optional: Plot des Histogramms mit Bin-Mittelpunkten als X-Achsenbeschriftung
    #plt.figure(figsize=(10, 6))
    #plt.bar(bin_centers, normalize_to_minus_one_and_one_v2(histogram), width=np.diff(bin_edges), edgecolor='black', align='center', alpha=0.75, color='blue')
    #plt.title('Dichteverteilung (64 Bins)')
    #plt.xlabel('Anzahl der Nachbarn (Dichte) - Bin-Mittelpunkte')
    #plt.ylabel('Relative Häufigkeit')
    #plt.grid(True)
    #plt.show()

    mean_distance = np.mean(normalize_to_minus_one_and_one_v2(distances))

    return normalize_to_minus_one_and_one_v2(histogram), mean_density, mean_distance, radius

def visualize_pcd_pcl(pcd_o3d=None, points = None, exe_path = os.path.join(os.getcwd(), "pcl_viewer_release.exe"), name="cloud"):
    if type(points) == type(None):
        # Save the point cloud as a PCD file
        o3d.io.write_point_cloud(name +".pcd", pcd_o3d)
    elif type(pcd_o3d) == type(None):
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud(name + ".pcd", pcd_o3d)
    else:
        print("set one of these parameters: pcd_o3d --> Pointcloud object of o3d or points --> Array")

    # Run the .exe file and capture the output
    result = subprocess.run(
        [exe_path, name + ".pcd"],  # Assuming executable prints result to stdout
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Check if the file exists
    if os.path.exists(name+".pcd"):
        os.remove(name+".pcd")

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
    print("num seg:", num_segments)

    # Initialize the list to store index groups
    index_groups = []

    # Iterate over segments and calculate indices for each segment
    for i in tqdm(range(num_segments), desc="Going through Segments..", ascii=True,  total=num_segments,dynamic_ncols=False):
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
    for i in tqdm(range(num_segments), desc="Going through Segments..",  total=num_segments,ascii=True, dynamic_ncols=False):
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
    for i in tqdm(range(num_segments), desc="Going through Segments..", total=num_segments, ascii=True, dynamic_ncols=False):
        # Calculate thresholds for the current segment
        segment_min = x_min + (i * x_percentage * x_range)
        segment_max = x_min + ((i + 1) * x_percentage * x_range)

        # Create a mask for points within the current segment
        mask = np.where((points[:, 0] >= segment_min) & (points[:, 0] < segment_max))[0]

        if len(mask) >= min_points:
        # Add the indices to the index groups list
            index_groups.append(mask.tolist())

    return index_groups

def normalize_to_minus_one_and_one_v2(descriptor):
    max_abs_val = np.max(np.abs(descriptor))

    # Avoid division by zero if max_abs_val is zero
    if max_abs_val == 0:
        return np.zeros_like(descriptor)  # If all values are zero, return zero vector

    # Normalize to range [-1, 1] while keeping zero centered
    normalized_descriptor = descriptor / max_abs_val
    return normalized_descriptor

def calculate_normals(pcd, k = 10, radius=None):
    if radius == None:
        radius, distances = calculate_mean_radius(pcd, k)

    # Normalschätzung für die Punktwolke durchführen
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))

    # Normalenvektoren abrufen
    normal_vectors = np.round(np.asarray(pcd.normals),2)

    return pcd, normal_vectors

def visualize_histogram(histogram):
    num_bins = len(histogram)
    bin_centers = np.arange(1, num_bins + 1)

    plt.figure(figsize=(8, 4))
    plt.bar(bin_centers, histogram, width=0.9, color='skyblue')
    plt.xlabel('Bin Number')
    plt.ylabel('Normalized Frequency')
    plt.title(f'Histogram - {num_bins} Bins')
    plt.ylim(0, np.max(histogram))  # Normalization from 0 to 1
    plt.xlim(1, num_bins)  # Set x-axis from 1 to 64
    plt.grid(True)
    plt.show()

def visualize_convex_hull(pcd, hull, points=None):
    if type(points) == type(None):
        # Save the point cloud as a PCD file
        pcd = pcd
        points = np.array(pcd.points)
    elif type(pcd) == type(None):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
    else:
        print("set one of these parameters: pcd_o3d --> Pointcloud object of o3d or points --> Array")
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the points
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='blue', s=50, marker='o')

    # Plot the convex hull
    for simplex in hull.simplices:
        ax.plot(points[simplex, 0], points[simplex, 1], points[simplex, 2], 'k-')

    # Create a Poly3DCollection to visualize the faces of the convex hull
    hull_faces = Poly3DCollection(points[hull.simplices], alpha=0.3, facecolor='cyan')
    ax.add_collection3d(hull_faces)

    # Set plot labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.title('3D Convex Hull')
    plt.show()

# histograms for Global histogram-based multi-feature descriptors
def compute_angle_histograms(pcd, normals = None, num_bins=64, k = 10, points = None, r = None, pca_based = True):
    if type(points) == type(None):
        # Save the point cloud as a PCD file
        pcd = pcd
        points = np.array(pcd.points)
    elif type(pcd) == type(None):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
    else:
        print("set one of these parameters: pcd_o3d --> Pointcloud object of o3d or points --> Array")

    if type(normals) == type(None):
        if type(r) == type(None):
            pcd, normals = calculate_normals(pcd, k=k)
        else:
            pcd, normals = calculate_normals(pcd, radius=r)

    # 4. Berechnung der Winkel der Normalenvektoren zu den sortierten Hauptachsen
    angles = np.zeros((len(normals), 3))

    if pca_based == True:
        # 1. PCA auf den Punkten durchführen
        pca = PCA(n_components=3)
        pca.fit(points)

        # 2. Eigenwerte und Hauptachsen extrahieren
        eigenvalues = pca.explained_variance_
        principal_axes = pca.components_

        # 3. Sortieren der Hauptachsen nach den Eigenwerten (absteigend)
        sorted_indices = np.argsort(eigenvalues)[::-1]  # Sortiere absteigend
        principal_axes = principal_axes[sorted_indices]

    else:
        principal_axes = [[1,0,0],[0,1,0],[0,0,1]]



    for i in range(3):  # Für jede Hauptachse
        # Berechne den Winkel zwischen jedem Normalenvektor und der aktuellen Hauptachse
        dot_product = np.dot(normals, principal_axes[i])
        norms_product = np.linalg.norm(normals, axis=1) * np.linalg.norm(principal_axes[i])
        cos_theta = dot_product / norms_product
        angles[:, i] = np.arccos(np.clip(cos_theta, -1.0, 1.0)) * 180 / np.pi  # Winkel in Grad

    # 5. Histogramme für die Winkel zu jeder der drei Hauptachsen erstellen
    histograms = []
    for i in range(3):
        histogram, _ = np.histogram(angles[:, i], bins=num_bins, range=(0, 360), density=True)
        histograms.append(normalize_to_minus_one_and_one_v2(histogram))

    return np.array(histograms)
def compute_distance_histogram(pcd, num_bins=64, points = None, pca_based=True):
    if type(points) == type(None):
        # Save the point cloud as a PCD file
        pcd = pcd
        points = np.array(pcd.points)
    elif type(pcd) == type(None):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
    else:
        print("set one of these parameters: pcd_o3d --> Pointcloud object of o3d or points --> Array")

    if pca_based == True:
        points_centered = points - np.mean(points, axis=0)  # Zentrieren der Punktwolke
    else:
        points_centered = points

    # 2. Abstände der zentrierten Punkte zum Ursprung des PCA-Raums berechnen
    distances = np.linalg.norm(points_centered, axis=1)

    # 3. Histogramm der Abstände erstellen
    histogram, _ = np.histogram(distances, bins=num_bins, range=(np.min(distances), np.max(distances)), density=True)

    return histogram

def compute_bounding_box_volume(pcd, points=None):
    if type(points) == type(None):
        # Save the point cloud as a PCD file
        pcd = pcd
        points = np.array(pcd.points)
    elif type(pcd) == type(None):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
    else:
        print("set one of these parameters: pcd_o3d --> Pointcloud object of o3d or points --> Array")
    # Find the minimum and maximum coordinates along each axis
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)

    # Calculate the lengths of the bounding box sides
    lengths = max_coords - min_coords

    # Volume is the product of the side lengths
    volume = np.prod(lengths)
    return volume

def compute_convex_hull_volume(pcd, points=None):
    if type(points) == type(None):
        # Save the point cloud as a PCD file
        pcd = pcd
        points = np.array(pcd.points)
    elif type(pcd) == type(None):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
    else:
        print("set one of these parameters: pcd_o3d --> Pointcloud object of o3d or points --> Array")
    # 1. Calculate the convex hull
    hull = ConvexHull(points)

    # 2. Get the volume of the convex hull
    volume = hull.volume

    return hull, volume

def compute_GHBMF(pcd, k=10, points = None): #global histogram-based multi-feature Descriptor
    if type(points) == type(None):
        # Save the point cloud as a PCD file
        pcd = pcd
        points = np.array(pcd.points)
    elif type(pcd) == type(None):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
    else:
        print("set one of these parameters: pcd_o3d --> Pointcloud object of o3d or points --> Array")


    # pose invariant
    histogram_distances_pca = compute_distance_histogram(pcd)
    histogram_density, mean_density, mean_distance, radius = calculate_density_histogram(pcd, k=k)
    histograms_angles_pca = np.concatenate(compute_angle_histograms(pcd, r=radius))
    hull, convex_hull_volume = compute_convex_hull_volume(pcd)

    #pose variant
    histogram_distances_global = compute_distance_histogram(pcd, pca_based= False)
    histograms_angles_global = np.concatenate(compute_angle_histograms(pcd, r=radius, pca_based=False))




    ghbmf_descriptor = np.concatenate([histograms_angles_global,histogram_distances_global,histograms_angles_pca, histogram_distances_pca, histogram_density, [mean_density], [mean_distance], [convex_hull_volume]])
    return ghbmf_descriptor


def compute_and_visualize_iss_features(pcd, point_cloud_file=None, non_max_radius=0.05, salient_radius=2):
    # Laden der Punktwolke
    if point_cloud_file == None:
        pcd = pcd
    else:
        pcd = o3d.io.read_point_cloud(point_cloud_file)

    # Berechnen der ISS-Merkmale

    iss_features = o3d.geometry.keypoint.compute_iss_keypoints(copy.deepcopy(pcd), salient_radius=salient_radius,
                                                               non_max_radius=non_max_radius)

    print(np.array(iss_features.points).shape)

    # Visualisierung der Punktwolken (Original und ISS-Merkmale)
    #o3d.visualization.draw_geometries(
    #    [pcd.paint_uniform_color([0, 0, 0]), iss_features.paint_uniform_color([255, 0, 0])])
    #o3d.visualization.draw_geometries([iss_features.paint_uniform_color([255, 0, 0])])
    return iss_features

def compute_3d_skeleton_LBC(pcd):
    from pc_skeletor import LBC

    lbc = LBC(point_cloud=pcd,
              down_sample=0.008)
    lbc.extract_skeleton()
    lbc.extract_topology()

    # Debug/Visualization
    lbc.visualize()
    lbc.export_results('./output')
    lbc.animate(init_rot=np.asarray([[1, 0, 0], [0, 0, 1], [0, 1, 0]]),
                steps=300,
                output='./output')

def compute_3d_skeleton_LBC(pcd):
    from pc_skeletor import SLBC

    s_lbc = SLBC(point_cloud={'trunk': pcd_trunk, 'branches': pcd_branch},
                 semantic_weighting=30,
                 down_sample=0.008,
                 debug=True)
    s_lbc.extract_skeleton()
    s_lbc.extract_topology()

    # Debug/Visualization
    s_lbc.visualize()
    s_lbc.show_graph(s_lbc.skeleton_graph)
    s_lbc.show_graph(s_lbc.topology_graph)
    s_lbc.export_results('./output')
    s_lbc.animate(init_rot=np.asarray([[1, 0, 0], [0, 0, 1], [0, 1, 0]]), steps=300, output='./output')


def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4559,
                                      front=[0.6452, -0.3036, -0.7011],
                                      lookat=[1.9892, 2.0208, 1.8945],
                                      up=[-0.2779, -0.9482, 0.1556])



def prepare_dataset(voxel_size):
    print(":: Load two point clouds and disturb initial pose.")

    source = o3d.io.read_point_cloud("bz3_cutted.ply")
    target = o3d.io.read_point_cloud("bz5_model.ply")
    trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    source.transform(trans_init)
    draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

    source_down = compute_and_visualize_iss_features(source_down)
    target_down = compute_and_visualize_iss_features(target_down)

    o3d.visualization.draw_geometries([source_down, target_down.paint_uniform_color([0,1,0])])
    return source, target, source_down, target_down, source_fpfh, target_fpfh

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result






# Beispielverwendung
if __name__ == "__main__":
    pcd1 = o3d.io.read_point_cloud("object_dachstuhl_rgb.ply")
    pcd2 = o3d.io.read_point_cloud("object_stahltraeger_rgb.ply")
    pcd3 = o3d.io.read_point_cloud("test_machine.ply")
    pcd4 = o3d.io.read_point_cloud("object_bulldozer_07.ply")
    pcd5 = o3d.io.read_point_cloud("2024-07-29_targetx_concrete_pm_pix4d_adjusted.ply")
    start = time.time()

    model = o3d.io.read_point_cloud("bz3_cutted.ply")

    # # Rotationsmatrix für 90 Grad um die Z-Achse erstellen
    # rotation_matrix = pcd3.get_rotation_matrix_from_xyz((np.pi / 4, 0, 0))
    # pcd_translate = copy.deepcopy(pcd4).translate((0, 0, 10))
    #
    # pcd_partial_id_lists = crop_group_point_cloud_by_percentage_x(copy.deepcopy(pcd4), 0.50)
    # pcd_partial = copy.deepcopy(pcd4).select_by_index(pcd_partial_id_lists[0])
    #
    # # Punktwolke rotieren
    # pcd_rot = copy.deepcopy(pcd4).rotate(rotation_matrix, center=pcd4.get_center())
    # ghbmf_desc = compute_GHBMF(pcd_partial, k=30)
    # print("desc time: ", time.time() - start)
    # visualize_histogram(ghbmf_desc[:-1])
    # visualize_histogram([ghbmf_desc[-1]])
    #
    #
    # # angles
    # start = time.time()
    # histograms_angles = compute_angle_histograms(pcd4, k=30)
    # print(histograms_angles.shape)
    # print(time.time()-start)
    # visualize_histogram(histograms_angles[0])
    # visualize_histogram(histograms_angles[1])
    # visualize_histogram(histograms_angles[2])
    #
    # # distances
    # start = time.time()
    # histogram_distances = compute_distance_histogram(pcd4)
    # visualize_histogram(histogram_distances)
    #
    # # density
    # start = time.time()
    # histogram_density, mean_density, mean_distance = calculate_density_histogram(pcd4, k = 30)
    # visualize_histogram(histogram_density)
    # print(histogram_density)
    # print(len(histogram_density))
    # print(time.time()-start)
    #
    # # Calculate Convex Hull Volume
    # hull, convex_hull_volume = compute_convex_hull_volume(pcd4)
    # print(convex_hull_volume)
    #

    voxel_size = 0.01  # means 5cm for this dataset
    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(voxel_size)
    result_ransac = execute_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                voxel_size)
    print(result_ransac)
    draw_registration_result(source_down, target_down, result_ransac.transformation)





    test_vfh_descriptor = False
    test_esf_descriptor = False
    test_harris3d_keypoints =False
    test_iss_keypoints = False
    test_3d_selecton = False

    if test_vfh_descriptor == True:
        # Rotationsmatrix für 90 Grad um die Z-Achse erstellen
        rotation_matrix = pcd3.get_rotation_matrix_from_xyz((0, np.pi / 2, 0))

        # Punktwolke rotieren
        pcd_rot= copy.deepcopy(pcd3).rotate(rotation_matrix, center=pcd3.get_center())
        vfh_descriptor = compute_vfh_vector(points=np.array(pcd1.points))
        print(vfh_descriptor)
        plot_histogram(vfh_descriptor, label = "ori")
        vfh_descriptor_rot = compute_vfh_vector(points=np.array(pcd_rot.points))
        print(vfh_descriptor_rot)
        plot_histogram(vfh_descriptor_rot, label = "rot")


    if test_esf_descriptor == True:
        # Rotationsmatrix für 90 Grad um die Z-Achse erstellen
        rotation_matrix = pcd4.get_rotation_matrix_from_xyz((np.pi / 2, 0, 0))


        # Punktwolke rotieren
        pcd_rot = copy.deepcopy(pcd4).rotate(rotation_matrix, center=pcd4.get_center())
        pcd_translate = copy.deepcopy(pcd4).translate((10,0,0))

        pcd_partial_id_lists = crop_group_point_cloud_by_percentage_z(copy.deepcopy(pcd4), 0.50)
        pcd_partial = copy.deepcopy(pcd4).select_by_index(pcd_partial_id_lists[0])


        #o3d.visualization.draw_geometries([pcd_rot, pcd4, pcd_translate, pcd_partial])

        #esf_descriptor = compute_esf_descriptor(points=np.array(pcd4.points))
        #print(esf_descriptor)
        #plot_histogram(normalize_to_minus_one_and_one_v2(esf_descriptor), label="ori")
        esf_descriptor_rot = compute_esf_descriptor(points=np.array(pcd_rot.points))
        print(esf_descriptor_rot)
        plot_histogram(normalize_to_minus_one_and_one_v2(esf_descriptor_rot), label="rot")

    if test_harris3d_keypoints == True:

        print(np.array(pcd4.points).shape)

        pcd_kp_harris3d = compute_harris3d_keypoints(pcd_o3d=model)
        kp_harris3d = np.array(pcd_kp_harris3d.points)
        print(kp_harris3d.shape)

        o3d.io.write_point_cloud("pcd_kp_harris3d_bz3.ply", pcd_kp_harris3d)

    if test_iss_keypoints == True:
        pcd_kp_iss = compute_and_visualize_iss_features(model, non_max_radius=0.05, salient_radius=2)
        o3d.io.write_point_cloud("pcd_kp_iss_bz3.ply", pcd_kp_iss)



