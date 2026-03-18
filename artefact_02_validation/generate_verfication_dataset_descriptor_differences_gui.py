import tkinter as tk
from tkinter import filedialog
import random
import sys
import gradio
import tempfile
from tqdm import tqdm
import open3d as o3d
import laspy as lp
import numpy as np
import copy
from matplotlib import pyplot as plt
from sklearn.neighbors import KDTree
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cosine
from scipy.stats import wasserstein_distance
import threading
import json
from threading import Thread
import os
import h5py
import torch
from torch.utils.data import Dataset
from flask import Flask, send_from_directory, request, jsonify
import gradio as gr

# directories to other folders in pipeline to get access to other data
dir_scripts     = os.getcwd()
dir_root        = os.path.normpath(os.getcwd() + os.sep + os.pardir)
dir_data        = os.path.join(dir_root, "data")

dir_data_3d_object_classification = os.path.join(dir_data, "3D-Object-Classification")
# data should be as txt format
dir_data_3d_object_classification_input_labeled = os.path.join(dir_data_3d_object_classification, "input_labeled")
# data should be in hdf5 format
dir_data_3d_object_classification_output_labeled = os.path.join(dir_data_3d_object_classification, "output_labeled")

dir_data_3d_segmentation = os.path.join(dir_data, "3D-Segmentation")
# data can be as txt, ply, las,... format
dir_data_3d_segmentation_input_unlabeled = os.path.join(dir_data_3d_segmentation, "input_unlabeled")
# data should be as txt format
dir_data_3d_segmentation_input_labeled= os.path.join(dir_data_3d_segmentation, "input_labeled")
dir_data_3d_segmentation_input_labeled_data = os.path.join(dir_data_3d_segmentation_input_labeled, "data")
dir_data_3d_segmentation_input_labeled_label = os.path.join(dir_data_3d_segmentation_input_labeled, "label")
# data should be in hdf5 format data:NxM (Amount_Points X Features[x_norm, y_norm, z_norm, x_normal, y_normal, z_normal, etc.]), label:Nx1 (Amount_Points X Label), LabelLegend->Labelname->label_number:1x1 (1 X Number), rgb_color:1x3 (1x(r,g,b))
dir_data_3d_segmentation_output_labeled = os.path.join(dir_data_3d_segmentation, "output_labeled")


#dir_data_3d_registration = os.path.join(dir_data, "3D-Registration")
#dir_data_3d_registration_input_labeled = os.path.join(dir_data_3d_registration, "input_labeled")
#list_data_pairs = os.listdir(dir_data_3d_registration_input_labeled)
#print(list_data_pairs)

dir_data_3d_verification= os.path.join(dir_data, "3D-Verification")
dir_data_3d_verification_input_data = os.path.join(dir_data_3d_verification, "input_data")
dir_data_3d_verification_input_data_planned_elements = os.path.join(dir_data_3d_verification_input_data, "planned_elements")
dir_data_3d_verification_input_data_scan= os.path.join(dir_data_3d_verification_input_data, "scan")
dir_data_3d_verification_output_dataset = os.path.join(dir_data_3d_verification, "output_dataset")
dir_data_3d_verification_temp = os.path.join(dir_data_3d_verification, "temp")



######################################################


def check_points_in_BoundingBox(pcd, bb):
    pcd_xyz = np.array(pcd.points)

    bb_xyz = np.array(bb.get_box_points())
    xmin_bb, xmax_bb = np.min(bb_xyz[:,0]) , np.max(bb_xyz[:,0])
    ymin_bb, ymax_bb = np.min(bb_xyz[:, 1]), np.max(bb_xyz[:, 1])
    zmin_bb, zmax_bb = np.min(bb_xyz[:, 2]), np.max(bb_xyz[:, 2])

    idxs_point_in_box = []
    for pt_idx, xyz in enumerate(pcd_xyz):
        x = xyz[0]
        y = xyz[1]
        z = xyz[2]
        is_point_in_box = ((xmin_bb <= x and xmax_bb >= x) and (ymin_bb <= y and ymax_bb >= y) and (zmin_bb <= z and zmax_bb >= z))
        if is_point_in_box == True:

            idxs_point_in_box.append(pt_idx)
    pcd_in_box = pcd.select_by_index(idxs_point_in_box)
    return pcd_in_box, idxs_point_in_box

def get_detected_element(pcd_as_planned_element, pcd_as_built_element_raw, k):
    if np.array(pcd_as_built_element_raw.points).shape[0] > k:
        # get points of as-planned and as_built state
        points_as_planned_element = np.array(pcd_as_planned_element.points)
        points_as_built_element = np.array(pcd_as_built_element_raw.points)
        print("As-Planned-Element: ", points_as_planned_element.shape)
        print("As-Built-Element: ", points_as_built_element.shape)

        #create search tree
        tree = KDTree(points_as_built_element)

        # get neighbours for each point in as-planned point cloud element which is in a specific radius
        idx_in_radius = []

        #indices, distances = tree.query_radius(X=points_as_planned_element, r=search_radius, return_distance=True, sort_results=True)
        distances, indices = tree.query(X=points_as_planned_element, k = k, return_distance=True, sort_results=True)

        first_write = True
        for point_idxs in indices:
            if first_write == True:
                idx_in_radius = point_idxs
                first_write = False
            else:
                idx_in_radius = np.concatenate((idx_in_radius, point_idxs), axis=0)

        point_to_save_idx = np.unique(idx_in_radius)
        # get only the point of the as_built state point cloud where the points are neighbours of the as_planned state point cloud
        pcd_as_built_element_detected = pcd_as_built_element_raw.select_by_index(point_to_save_idx)
        return pcd_as_built_element_detected, point_to_save_idx
    else:
        return pcd_as_built_element_raw, []

def calculate_esf_descriptor_difference(pcd_as_planned_element, pcd_scan_in_box):
    if np.array(pcd_scan_in_box.points).shape[0] > 0:
        esf_planned = pc_desc.compute_esf_descriptor(pcd_o3d=pcd_as_planned_element)
        esf_scan = pc_desc.compute_esf_descriptor(pcd_o3d=pcd_as_built_on_surface)
        print("Planned ESF:")
        print(esf_planned)

        print("Scan ESF:")
        print(esf_scan)

        print("Diff ESF:")
        esf_diff = np.array(esf_planned)-np.array(esf_scan)

        print(len(esf_diff))
        print(esf_diff)
        return np.array(esf_diff)
    else:
        return np.zeros_like((1,640))

def normalize_to_minus_one_and_one_v2(descriptor):
    max_abs_val = np.max(np.abs(descriptor))

    # Avoid division by zero if max_abs_val is zero
    if max_abs_val == 0:
        return np.zeros_like(descriptor)  # If all values are zero, return zero vector

    # Normalize to range [-1, 1] while keeping zero centered
    normalized_descriptor = descriptor / max_abs_val
    return normalized_descriptor

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

def add_idx_lists_0_5_to_dict(pcd_element_current, pcd_as_built_on_surface,  dict_dataset_esf_desc_diff, file_key, post_fix, label = 1, esf_planned = None):
    if esf_planned == None:
        esf_planned = pc_desc.compute_esf_descriptor(pcd_o3d=pcd_element_current)

    esf_scan = pc_desc.compute_esf_descriptor(pcd_o3d=pcd_as_built_on_surface)
    if esf_scan == []:
        return dict_dataset_esf_desc_diff

    if np.array(esf_scan).shape[0] == 640:

        esf_diff = np.array(esf_planned) - np.array(esf_scan)
        esf_scan_diff_norm = normalize_to_minus_one_and_one_v2(esf_diff)

        emd_values = compute_emd_for_esf(esf_planned, esf_scan)
        cosine_similarity_values = calculate_cosine_similarity(esf_planned, esf_scan)

        esf_diff_extended_norm = np.concatenate((esf_scan_diff_norm, emd_values, cosine_similarity_values))

        esf_diff_extended_norm = np.round(esf_diff_extended_norm, decimals=4)

        dict_dataset_esf_desc_diff[file_key.replace(".ply", "label" + str(label) +"_part_"+ post_fix)] = {"esf_diff": list(esf_diff_extended_norm), "label": label}

    else:
        return dict_dataset_esf_desc_diff

    return dict_dataset_esf_desc_diff

def compare_objects_for_verification(pcd_as_planned_element, pcd_scan_in_box, k_nearest_neighbors, use_refinement=True):
    if np.array(pcd_scan_in_box.points).shape[0] > 0:
        # verfeinerung sodass nicht alle Punkte in der Boundingbox sondern nur die nächsten zur geplanten Oberfläche genommen werden
        if use_refinement == True:
            pcd_as_built_on_surface, point_to_save_idx = get_detected_element(pcd_as_planned_element=pcd_as_planned_element, pcd_as_built_element_raw=pcd_scan_in_box, k=k_nearest_neighbors)
        else:
            pcd_as_built_on_surface = pcd_scan_in_box

        esf_planned = pc_desc.compute_esf_descriptor(pcd_o3d=pcd_as_planned_element)
        esf_scan = pc_desc.compute_esf_descriptor(pcd_o3d=pcd_as_built_on_surface)
        print("Planned ESF:")
        print(esf_planned)

        print("Scan ESF:")
        print(esf_scan)

        print("Diff ESF:")
        esf_diff = esf_planned-esf_scan

        print(len(esf_diff))
        print(esf_diff)


        return esf_diff
    else:
        return []

def calculate_expansion_value(bounding_box, expansion_fraction):
    # Berechne die Dimensionen der Bounding Box
    min_bound = np.array(bounding_box.min_bound)
    max_bound = np.array(bounding_box.max_bound)
    dimensions = max_bound - min_bound

    # Berechne den Expansion-Wert als Prozentsatz der Dimensionen
    expansion_value = dimensions * expansion_fraction

    return expansion_value


def expand_bounding_box(pcd, expansion_fraction):
    # Berechne die Bounding Box der Punktwolke
    bounding_box = pcd.get_axis_aligned_bounding_box()

    # Berechne den Expansion-Wert
    expansion_value = calculate_expansion_value(bounding_box, expansion_fraction)

    # Erweitere die Bounding Box entlang jeder Achse
    expanded_min_bound = np.array(bounding_box.min_bound) - expansion_value
    expanded_max_bound = np.array(bounding_box.max_bound) + expansion_value

    # Erzeuge eine neue Bounding Box mit den erweiterten Werten
    expanded_bounding_box = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=expanded_min_bound,
        max_bound=expanded_max_bound
    )

    return expanded_bounding_box


def visualize_and_input(pcd1, pcd2):
    # Create visualizers
    vis1 = o3d.visualization.Visualizer()
    vis2 = o3d.visualization.Visualizer()

    # Initialize visualizers
    vis1.create_window(window_name="Point Cloud 1", width=800, height=600)
    vis2.create_window(window_name="Point Cloud 2", width=800, height=600)

    # Add point clouds to visualizers
    vis1.add_geometry(pcd1)
    vis2.add_geometry(pcd2)

    # Run visualizers in separate threads
    vis1.run()
    vis2.run()

    # Prompt user for input
    while True:
        try:
            user_input = int(input("Enter 0, 1, or 2: "))
            if user_input in [0, 1, 2]:
                print(f"You entered {user_input}.")
                break
            else:
                print("Invalid input. Please enter 0, 1, or 2.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def normalize_to_minus_one_and_one(descriptor):
    min_val = np.min(descriptor)
    max_val = np.max(descriptor)
    normalized_descriptor = 2 * (descriptor - min_val) / (max_val - min_val) - 1
    return normalized_descriptor


def plot_histogram(descriptor, label):
    plt.figure(figsize=(10, 5))

    # Each bin corresponds to one value in the 640-dimensional vector
    plt.bar(range(len(descriptor)), descriptor, width=1.0, edgecolor='black')

    plt.xlabel('Vector Index (0 to 639)')
    plt.ylabel('Normalized Value (-1 to 1)')
    plt.title('Histogram of ESF Descriptor Differences - ' + label)

    plt.ylim(-1, 1)  # Set the y-axis limits to -1 to 1
    plt.xlim(0, len(descriptor))  # Set the x-axis limits from 0 to 640

    plt.show()



# gui
# read pointcloud in any format:
def read_pcd_in_any_format(dir_pcd_path):
    file_extension = os.path.splitext(dir_pcd_path)[1]
    point_cloud = o3d.geometry.PointCloud()

    if file_extension == ".txt":
        data_txt = np.loadtxt(dir_pcd_path, delimiter=",")
        if data_txt.shape[1] == 3:
            point_cloud.points = o3d.utility.Vector3dVector(data_txt)
        elif data_txt.shape[1] == 4:  # x,y,z,intensity
            point_cloud.points = o3d.utility.Vector3dVector(data_txt[:3])
        elif data_txt.shape[1] == 6:  # x,y,z,r,g,b
            point_cloud.points = o3d.utility.Vector3dVector(data_txt[:3])
        elif data_txt.shape[1] == 7:  # x,y,z,r,g,b,intensity
            point_cloud.points = o3d.utility.Vector3dVector(data_txt[:3])
    elif file_extension == ".las":
        lasdata = lp.read(dir_pcd_path)
        print(lasdata)
        data_las = []
        # point values with plydata
        x = np.array(lasdata.x)
        x = x.reshape((len(x), 1))
        y = np.array(lasdata.y)
        y = y.reshape((len(y), 1))
        z = np.array(lasdata.z)
        z = z.reshape((len(z), 1))
        data_las = np.concatenate((x, y, z), axis=1)
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(data_las)
        try:
            # Combine the RGB color components into a list of RGB tuples
            rgb_list = [[r, g, b] for r, g, b in zip(lasdata.red, lasdata.green, lasdata.blue)]
            point_cloud.colors = o3d.utility.Vector3dVector(np.array(rgb_list))
        except:
            num_points = x.shape[0]
            rgb_list = [[r, g, b] for r, g, b in zip(np.zeros(num_points), np.zeros(num_points), np.zeros(num_points))]
            point_cloud.colors = o3d.utility.Vector3dVector(np.array(rgb_list))

    elif file_extension == ".ply":
        point_cloud = o3d.io.read_point_cloud(dir_pcd_path)

    elif file_extension == ".asc":
        asc_pts_list = []
        asc_rgb_list = []
        with open(dir_pcd_path, 'r') as f:
            lines = np.array(f.readlines())
            for line in lines:
                s = line.split(",")
                if s.shape[1] == 3:
                    asc_pts_list.append([float(s[0]), float(s[1]), float(s[2])])

                elif s.shape[1] == 4:  # x,y,z,intensity
                    asc_pts_list.append([float(s[0]), float(s[1]), float(s[2])])
                    asc_rgb_list.append([float(0), float(0), float(0)])

                elif s.shape[1] == 6:  # x,y,z,r,g,b
                    asc_pts_list.append([float(s[0]), float(s[1]), float(s[2])])
                    asc_rgb_list.append([float(s[3]), float(s[4]), float(s[5])])

                elif s.shape[1] == 7:  # x,y,z,r,g,b,intensity
                    asc_pts_list.append([float(s[0]), float(s[1]), float(s[2])])
                    asc_rgb_list.append([float(s[3]), float(s[4]), float(s[5])])

            point_cloud.points = o3d.utility.Vector3dVector(np.array(asc_pts_list))
            point_cloud.colors = o3d.utility.Vector3dVector(np.array(asc_rgb_list))

    elif file_extension in [".obj", ".stl", ".off", "gltf"]:
        point_cloud = convert_obj_to_ply_poisson_disk(dir_pcd_path, sampling_points=4096)

    return point_cloud


def convert_obj_to_ply_poisson_disk(dir_path, sampling_points=4096):
    print("Start Converting .obj in fixed points Pointclouds..")
    # HYPERPARAMETER
    mesh = o3d.io.read_triangle_mesh(dir_path)
    pcd = mesh.sample_points_poisson_disk(sampling_points)
    return pcd


def generate_viewer_html(viewer_id, points=None, colors=None, point_size=0.1):
    if points is None or colors is None:
        # Falls keine Punktwolke übergeben wurde, eine leere Szene generieren
        points = []
        colors = []

    points_json = json.dumps(points)
    colors_json = json.dumps(colors)

    points_array = np.array(points).reshape(-1, 3) if points else np.empty((0, 3))

    print(points_array.shape)
    if points_array.shape[0] > 0:
        mean_x, mean_y, mean_z = np.mean(np.round(points_array, 3), axis=0)
        max_x, max_y, max_z = np.max(points_array, axis=0)
    else:
        mean_x, mean_y, mean_z, max_x, max_y, max_z = 0, 0, 0, 0, 0, 0

    js_code = f"""
        var scene = new THREE.Scene();
        scene.background = new THREE.Color(0xffffff); // Set background to white
        var camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 1, 1000);
        camera.position.set(0, 0, 10);

        var renderer = new THREE.WebGLRenderer({{ canvas: document.getElementById('myCanvas'), antialias: true}});
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio( window.devicePixelRatio );
        window.addEventListener('resize', function() {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }});

        // TrackballControls für die Interaktion
        var controls = new THREE.TrackballControls(camera, renderer.domElement);
        controls.dynamicDampingFactor = 0.1;
        controls.rotateSpeed = 2.0;
        controls.zoomSpeed = 0.2;
        controls.panSpeed = 1.0;
        controls.noZoom = false;
        controls.noPan = false;
        controls.update();



        var light = new THREE.DirectionalLight(0xffffff, 0.5);
        light.position.setScalar(1);
        scene.add(light, new THREE.AmbientLight(0xffffff, 0.5));

        // Add axes helper to show the coordinate system
        var axesHelper = new THREE.AxesHelper(5);
        scene.add(axesHelper);

        // Add grid helper for XY plane
        var grid = new THREE.GridHelper(20, 20);
        grid.rotation.x = Math.PI / 2;  // Rotate the grid to lie in the XY plane
        scene.add(grid);

        // Punktwolke laden
        var geometry = new THREE.BufferGeometry();
        var positions = new Float32Array({points_json});
        var colors = new Float32Array({colors_json});

        geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

        var material = new THREE.PointsMaterial({{size: {point_size}, vertexColors: true, transparent: false, depthWrite: true, sizeAttenuation: true }});
        material.sizeAttenuation = true;
        var points = new THREE.Points(geometry, material);
        scene.add(points);

        function animate() {{
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);

        }}

        animate();

        // Raycasting für Punkt-Auswahl
    document.addEventListener('click', function(event) {{
        var mouse = new THREE.Vector2();
        var raycaster = new THREE.Raycaster();
        var rect = renderer.domElement.getBoundingClientRect();

        mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

        raycaster.setFromCamera(mouse, camera);
        var intersects = raycaster.intersectObject(points);

        if (intersects.length > 0) {{
            var point = intersects[0].point;
            console.log('Clicked point coordinates:', point.x, point.y, point.z);
            document.getElementById('referencePoint').textContent = 
                'Referenzpunkt: x=' + point.x.toFixed(2) + ', y=' + point.y.toFixed(2) + ', z=' + point.z.toFixed(2);
        }}
        }});

        // Funktion zum Aktualisieren der Punktgröße
        var pointSizeDisplay = document.getElementById('pointSizeValue');
        document.getElementById('pointSizeInput').addEventListener('input', function(event) {{
        var newSize = parseFloat(event.target.value);
        points.material.size = newSize;
        points.material.needsUpdate = true;  // Aktualisiert das Material der Punkte
        pointSizeDisplay.textContent = 'Point Size: ' + newSize.toFixed(2);
        }});

        // Reset-Knopf Funktion
        document.getElementById('resetButton').addEventListener('click', function() {{
            // Setze Referenzpunkt auf den Mittelwert
            var meanX = {mean_x};
            var meanY = {mean_y};
            var meanZ = {mean_z};

            document.getElementById('referencePoint').textContent = 
                'Referenzpunkt (Mean): x=' + meanX.toFixed(2) + ', y=' + meanY.toFixed(2) + ', z=' + meanZ.toFixed(2);

            // Optional: Du kannst auch visuelle Hinweise hinzufügen, z.B. den Mean-Punkt markieren
        }});
        """

    html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Viewer {viewer_id}</title>
            <style>
                body {{
                    margin: 0;
                    overflow: hidden; /* Entfernt Scrollbars */
                }}
                canvas {{
                    display: block;
                }}
                #controls {{
                    position: absolute;
                    top: 10px;
                    left: 10px;
                    z-index: 1;
                    font-family: Arial, sans-serif;
                }}
                #referencePoint {{
                    font-size: 12px;
                    background-color: rgba(255, 255, 255, 0.8);
                    padding: 5px;
                    border-radius: 5px;
                    margin-top: 10px;
                    display: inline-block;
                }}
                #pointSizeValue {{
                    margin-left: 10px;
                    font-size: 12px;
                }}
            </style>
        </head>
        <body>
            <div id="controls">
                <label for="pointSizeInput">Point Size: </label>
                <input type="range" id="pointSizeInput" min="0.01" max="1.0" step="0.01" value="{point_size}">
                 <span id="pointSizeValue">Point Size: {point_size}</span>
                <button id="resetButton">Reset Referenzpunkt</button>
                <p id="referencePoint">Referenzpunkt: x={mean_x}, y={mean_y}, z={mean_z}</p>
            </div>
            <canvas id="myCanvas"></canvas>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/TrackballControls.js"></script>

            <script>
            {js_code}
            </script>
        </body>
        </html>
        """

    with open(os.path.join(HTML_DIR, f'viewer_{viewer_id}.html'), 'w') as f:
        f.write(html_content)


# Gradio-Interface erstellen
def get_base64_image(image_path):
    import base64
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

def random_uniform_downsampling (pcd= None, points = None, colors = None, min_points = 1024):
    """
    Randomly select a specified percentage of points from a point cloud.

    Parameters:
    - point_cloud: Open3D point cloud object
    - percentage: Percentage of points to randomly select (value between 0 and 1)

    Returns:
    - selected_indices: Indices of the randomly selected points
    """
    # Get the number of points in the point cloud
    if type(points) == type(None):
        num_points = len(pcd.points)
    else:
        num_points = points.shape[0]

    # Calculate the number of points to select based on the percentage
    num_points_to_select = min_points

    # Generate random indices for selecting points
    random_indices = np.random.choice(num_points, num_points_to_select, replace=False)

    if type(points) == type(None):
        colors = np.array(pcd.colors)
        pcd_downsampled = pcd.select_by_index(random_indices)
        pcd_downsampled.colors = o3d.utility.Vector3dVector(colors)
    else:
        pcd_downsampled= o3d.geometry.PointCloud()
        pcd_downsampled.points = o3d.utility.Vector3dVector(points[random_indices])
        if type(colors) != type(None):
            pcd_downsampled.colors = o3d.utility.Vector3dVector(colors)


    return pcd_downsampled
# Funktion zum Laden der Punktwolke und Erstellen von HTML-Dateien
def load_point_cloud(file):
    global pcd_input, pcd_viewer1, port, pcd_input_down
    pcd_input = read_pcd_in_any_format(file.name)
    pcd_input_viewer = copy.deepcopy(pcd_input)
    pcd_input_viewer = pcd_input_viewer.translate((-1) * np.mean(np.array(pcd_input_viewer.points), axis=0))
    if not pcd_input_viewer.has_colors():
        pcd_input_viewer.paint_uniform_color([0.5, 0.5, 0.5])


    desired_num_points = 2000000
    if np.array(pcd_input_viewer.points).shape[0] > desired_num_points:
        #pcd_viewer1 = pcd_input.farthest_point_down_sample(desired_num_points)
        pcd_viewer1 = random_uniform_downsampling(pcd = pcd_input_viewer, min_points= desired_num_points)
        pcd_input_down = random_uniform_downsampling(pcd = pcd_input, min_points= desired_num_points)
    else:
        pcd_viewer1 = pcd_input_viewer
        pcd_input_down = pcd_input




    points = np.round(np.array(pcd_viewer1.points),3)  # -np.mean(np.array(pcd.points))
    points = points.flatten().tolist()
    colors = np.asarray(pcd_viewer1.colors).flatten().tolist()

    # Generiere HTML-Datei für Viewer 1


    cache_buster = random.randint(1, 10000)

    generate_viewer_html(1, points, colors)

    # Viewer 2 bleibt leer, bis das Dropdown geändert wird
    #generate_viewer_html(2)

    # Rückgabe des iframe für Viewer 1 (Viewer 2 bleibt leer)
    return f'<iframe src="http://localhost:{port}/viewer_1.html?nocache={cache_buster}" style="width: 100%; height: 500px; border: none;"></iframe>'

def load_pcd_model(file):
    global pcd_model
    pcd_model = read_pcd_in_any_format(file.name)


def scale_pcd_to_input_range(pcd_input, pcd_proj):
    # Berechne die maximale Range der Eingabe-Punktwolke (pcd_input)
    pcd_input_points = np.array(pcd_input.points)
    max_range_input = np.max(pcd_input_points, axis=0) - np.min(pcd_input_points, axis=0)
    max_range_value = np.max(max_range_input)  # Maximaler Wert der Range

    # Definiere die Punktwolken, die skaliert werden sollen

    pcd_points = np.array(pcd_proj.points)
    # Berechne den Skalierungsfaktor
    scale_factor = max_range_value / np.max(np.ptp(pcd_points, axis=0))
    # Skaliere die Punkte
    pcd_proj.points = o3d.utility.Vector3dVector(pcd_points * scale_factor)

    return pcd_proj
# Funktion zum Generieren der Punktwolke für Viewer 2 bei Dropdown-Auswahl

def update_viewer_1_ai(as_built_segmented):
    global port
    pcd_part_viewer_1 = copy.deepcopy(as_built_segmented).translate((-1) * np.mean(np.array(copy.deepcopy(as_built_segmented).points), axis=0))
    points = np.array(pcd_part_viewer_1.points)
    colors = np.array(pcd_part_viewer_1.colors)

    # Konvertiere die Arrays in eine flache Liste für JSON
    points_list = points.flatten().tolist()
    colors_list = colors.flatten().tolist()
    cache_buster = random.randint(1, 10000)

    generate_viewer_html(1, points_list, colors_list)

    return f'<iframe src="http://localhost:{port}/viewer_1.html?nocache={cache_buster}" style="width: 100%; height: 500px; border: none;"></iframe>'

def update_viewer_3(pcd_part):
    global port
    pcd_part_viewer_2 = copy.deepcopy(pcd_part).translate((-1) * np.mean(np.array(copy.deepcopy(pcd_part).points), axis=0))
    points = np.array(pcd_part_viewer_2.points)
    colors = np.array(pcd_part_viewer_2.colors)

    # Konvertiere die Arrays in eine flache Liste für JSON
    points_list = points.flatten().tolist()
    colors_list = colors.flatten().tolist()
    cache_buster = random.randint(1, 10000)

    generate_viewer_html(3, points_list, colors_list)

    return f'<iframe src="http://localhost:{port}/viewer_3.html?nocache={cache_buster}" style="width: 100%; height: 500px; border: none;"></iframe>'

def update_viewer_2(pcd_part, pcd_scan):

    merged_pcd = pcd_scan.paint_uniform_color([0,1,0]) + pcd_part.paint_uniform_color([0.5,0.5,0.5])
    merged_pcd = merged_pcd .translate((-1) * np.mean(np.array(merged_pcd.points), axis=0))
    points = np.array(merged_pcd.points)
    colors = np.array(merged_pcd.colors)
    print(np.unique(colors))

    # Konvertiere die Arrays in eine flache Liste für JSON
    points_list = points.flatten().tolist()
    colors_list = colors.flatten().tolist()
    cache_buster = random.randint(1, 10000)

    generate_viewer_html(2, points_list, colors_list)

    return f'<iframe src="http://localhost:{port}/viewer_2.html?nocache={cache_buster}" style="width: 100%; height: 500px; border: none;"></iframe>'

def update_viewer_4(pcd_part, pcd_part_scan):

    merged_pcd = pcd_part_scan.paint_uniform_color([0,1,0]) + pcd_part.paint_uniform_color([0.5,0.5,0.5])
    merged_pcd = merged_pcd .translate((-1) * np.mean(np.array(merged_pcd.points), axis=0))
    points = np.array(merged_pcd.points)
    colors = np.array(merged_pcd.colors)
    print(np.unique(colors))

    # Konvertiere die Arrays in eine flache Liste für JSON
    points_list = points.flatten().tolist()
    colors_list = colors.flatten().tolist()
    cache_buster = random.randint(1, 10000)

    generate_viewer_html(4, points_list, colors_list)

    return f'<iframe src="http://localhost:{port}/viewer_4.html?nocache={cache_buster}" style="width: 100%; height: 500px; border: none;"></iframe>'

def update_viewer_5(pcd_part_scan):

    merged_pcd = copy.deepcopy(pcd_part_scan)
    merged_pcd = merged_pcd.translate((-1) * np.mean(np.array(merged_pcd.points), axis=0))
    points = np.array(merged_pcd.points)
    colors = np.array(merged_pcd.colors)
    print(np.unique(colors))

    # Konvertiere die Arrays in eine flache Liste für JSON
    points_list = points.flatten().tolist()
    colors_list = colors.flatten().tolist()
    cache_buster = random.randint(1, 10000)

    generate_viewer_html(5, points_list, colors_list)

    return f'<iframe src="http://localhost:{port}/viewer_5.html?nocache={cache_buster}" style="width: 100%; height: 500px; border: none;"></iframe>'

def show_pcd_in_o3d_viewer1():
    global pcd_input, pcd_viewer1, pcd_viewer2
    o3d.visualization.draw_geometries([pcd_viewer1])

def show_pcd_in_o3d_viewer2():
    global pcd_input, pcd_viewer1, pcd_viewer2
    o3d.visualization.draw_geometries([pcd_viewer2])

# Step 1: Create a function to save the .ply file with user-selected directory
def save_ply_file(pcd):
    # Initialize tkinter root
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Open a file dialog to select where to save the .ply file
    file_path = filedialog.asksaveasfilename(defaultextension=".ply",
                                             filetypes=[("PLY files", "*.ply")],
                                             title="Save PLY file")

def compare():
    global assembly_parts_folder, part_id, pcd_part, pcd_as_built_on_surface, pcd_input_down, pcd_model
    if len(assembly_parts_folder) > 0:
        part_files = os.listdir(assembly_parts_folder)
        part_files = [f for f in part_files if f.lower().endswith(".ply")]
        if part_files[part_id].endswith(".ply"):
            pcd_as_built_on_surface = process_code(pcd_input, pcd_part)

            viewer_3_update = update_viewer_3(pcd_part)
            viewer_4_update = update_viewer_4(pcd_part, pcd_as_built_on_surface)
            viewer_5_update = update_viewer_5(pcd_as_built_on_surface)
            return viewer_3_update, viewer_4_update, viewer_5_update, gr.Textbox(label=f"Antwort", interactive=False)
def load_part():
    global pcd_input, assembly_parts_folder, part_id, pcd_part, pcd_as_built_on_surface, part_file_name, pcd_input_down
    if len(assembly_parts_folder) > 0:
        part_files = os.listdir(assembly_parts_folder)
        part_files= [f for f in part_files if f.lower().endswith(".ply")]
        print(part_files)
        print(part_id)
        if part_files[part_id].endswith(".ply"):
            print(part_files[part_id])
            current_file = os.path.join(assembly_parts_folder, part_files[part_id])

            part_file_name = part_files[part_id]
            pcd_part = read_pcd_in_any_format(current_file)

            viewer_2_update = update_viewer_2(pcd_part, pcd_input_down)

            return viewer_2_update, gr.HTML(f"""<h3 style="margin-top: 5px; margin-bottom: 5px;">As-Planned Part ID: {part_id+1} - {part_file_name}</h3> """), gr.HTML(f"""<h3 style="margin-top: 5px; margin-bottom: 5px;">As-Built vs As-Planned Part ID: {part_id+1} - {part_file_name}</h3> """), gr.Textbox(label=f"Dateinamenanhang pro Label Scan - {part_file_name.replace('.ply', 'label0-1_part_post_fix')} # postfix e.g. manuel_ma_bz7 '_partid' will be added automatically ")


def load_next_part():
    global assembly_parts_folder, part_id, pcd_part, part_file_name

    print("Part id: ", part_id +1)

    if len(assembly_parts_folder) > 0:
        part_files = os.listdir(assembly_parts_folder)
        part_files = [f for f in part_files if f.lower().endswith(".ply")]
        if part_files[part_id+1].endswith(".ply"):

            current_file = os.path.join(assembly_parts_folder, part_files[part_id+1])
            part_file_name = part_files[part_id+1]
            pcd_part = read_pcd_in_any_format(current_file)

            part_id = part_id+1

            viewer_2_update = update_viewer_2(pcd_part, pcd_input_down)

            return viewer_2_update, gr.HTML(f"""<h3 style="margin-top: 5px; margin-bottom: 5px;">As-Planned Part ID: {part_id+1} - {part_file_name} </h3> """), gr.HTML(f"""<h3 style="margin-top: 5px; margin-bottom: 5px;">As-Built vs As-Planned Part ID: {part_id+1} - {part_file_name}</h3> """), gr.Textbox(label=f"Dateinamenanhang pro Label Scan - {part_file_name.replace('.ply', 'label0-1_part_post_fix')} # postfix e.g. manuel_ma_bz7 '_partid' will be added automatically ")
def load_previous_part():
    global assembly_parts_folder, part_id, pcd_part, part_file_name, pcd_input_down
    if len(assembly_parts_folder) > 0:
        part_files = os.listdir(assembly_parts_folder)
        part_files = [f for f in part_files if f.lower().endswith(".ply")]

        if part_id > 0 and part_files[part_id-1].endswith(".ply"):
            print("Part id: ", part_id - 1)
            current_file = os.path.join(assembly_parts_folder, part_files[part_id-1])
            part_file_name = part_files[part_id-1]
            pcd_part = read_pcd_in_any_format(current_file)
            viewer_update = update_viewer_2(pcd_part, pcd_input_down)
            part_id = part_id -1
            return viewer_update, gr.HTML(f"""<h3 style="margin-top: 5px; margin-bottom: 5px;">As-Planned Part ID: {part_id+1} - {part_file_name}</h3> """), gr.HTML(f"""<h3 style="margin-top: 5px; margin-bottom: 5px;">As-Built vs As-Planned Part ID: {part_id+1} - {part_file_name}</h3> """),  gr.Textbox(label=f"Dateinamenanhang pro Label Scan - {part_file_name.replace('.ply', 'label0-1_part_post_fix')} # postfix e.g. manuel_ma_bz7 '_partid' will be added automatically ")

def save_labeled_data_in_dict(pcd_part_file_name, post_fix, label):
    global pcd_input, port, pcd_part, pcd_as_built_on_surface, part_file_name

    dict_dataset_esf_desc_diff = {}
    json_file_name = "dataset_obj_verification_esf_extended_part3_manuel_labeled.json"

    try:
        with open(json_file_name) as f:
            dict_dataset_esf_desc_diff = json.load((f))
            print(len(list(dict_dataset_esf_desc_diff.keys())))

        data = list(dict_dataset_esf_desc_diff.keys())
        skip = True
    except:
        print("Failed")
        skip = False

    if skip == True:
        name_0 = part_file_name.replace(".ply", "label" + str(0) + "_part_" + post_fix)
        name_1 = part_file_name.replace(".ply", "label" + str(1) + "_part_" + post_fix)
        name_2 = part_file_name.replace(".ply", "label" + str(2) + "_part_" + post_fix)
        if name_0 in data or name_1 in data or name_2 in data:
            print("Already there: ", part_file_name)
            return gr.Textbox(label=f"Already Saved {part_file_name}", interactive=False)


        else:
            file_key = part_file_name

            # Visualize the two Point Cloud two assign a label to the data set by manual observation 0: not there, 1: partially there, 2: fully there
            # visualize_and_input(pcd_element_current, pcd_as_built_on_surface)

            dict_dataset_esf_desc_diff = add_idx_lists_0_5_to_dict(pcd_element_current=pcd_part,
                                                                   pcd_as_built_on_surface=pcd_as_built_on_surface,
                                                                   dict_dataset_esf_desc_diff=dict_dataset_esf_desc_diff,
                                                                   file_key=file_key,
                                                                   post_fix=post_fix, label=label)

            with open(json_file_name, 'w') as json_file:
                json.dump(dict_dataset_esf_desc_diff, json_file, indent=4)

            print("Saved Part:  ", file_key)
            return gr.Textbox(label=f"Saved {file_key}", interactive=False)
    else:
        file_key = part_file_name

        # Visualize the two Point Cloud two assign a label to the data set by manual observation 0: not there, 1: partially there, 2: fully there
        # visualize_and_input(pcd_element_current, pcd_as_built_on_surface)

        dict_dataset_esf_desc_diff = add_idx_lists_0_5_to_dict(pcd_element_current=pcd_part,
                                                               pcd_as_built_on_surface=pcd_as_built_on_surface,
                                                               dict_dataset_esf_desc_diff=dict_dataset_esf_desc_diff,
                                                               file_key=file_key,
                                                               post_fix=post_fix, label=label)

        with open(json_file_name, 'w') as json_file:
            json.dump(dict_dataset_esf_desc_diff, json_file, indent=4)

        print("Saved Part:  ", file_key)
        return gr.Textbox(label=f"Saved {file_key}", interactive=False)


def process_code(pcd_input, pcd_part, num_classes = 2, selected_class = 0, use_ai=False):
    global as_built_segmented
    # HYPERPARAMETER
    pcd_scan =  pcd_input
    expansion_fraction = 0.1
    use_refinement = True
    k_nearest_neighbors = 1

    pcd_element_current = pcd_part
    bb_as_planned_scaled = expand_bounding_box(copy.deepcopy(pcd_element_current), expansion_fraction)
    pcd_as_built_in_box, idxs_point_in_box = check_points_in_BoundingBox(pcd_scan, bb_as_planned_scaled)

    # verfeinerung sodass nicht alle Punkte in der Boundingbox sondern nur die nächsten zur geplanten Oberfläche genommen werden
    if use_refinement == True:
        pcd_as_built_on_surface, point_to_save_idx = get_detected_element(
            pcd_as_planned_element=pcd_element_current, pcd_as_built_element_raw=pcd_as_built_in_box,
            k=k_nearest_neighbors)
    else:
        pcd_as_built_on_surface = pcd_as_built_in_box

    return pcd_as_built_on_surface

def get_folder_dir(file_in_path):
    global assembly_parts_folder

    assembly_parts_folder = file_in_path
    print(file_in_path)
    print("Absoluter Pfad:", assembly_parts_folder)
    print("Amount Parts:", os.listdir(assembly_parts_folder))



def export_pointcloud_for_download():
    global pcd_as_built_on_surface, part_file_name
    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, "as_built_" +part_file_name)
    o3d.io.write_point_cloud(tmp_path, pcd_as_built_on_surface)
    return tmp_path


def build_interface():
    logo_path = os.path.join(dir_scripts, "logo")
    logo_file_path = get_base64_image(os.path.abspath(os.path.join(logo_path, "pcd_projection.png")))
    global pcd_input, pcd_viewer1, pcd_viewer2
    global assembly_parts_folder
    with gr.Blocks() as demo:
        gr.HTML(f"""
            <div style="text-align: center; padding: 20px;">
                <div style="display: flex; flex-direction: column; align-items: center;">
                <img src="data:image/png;base64,{logo_file_path}" alt="Logo" style="width: 200px; height: auto; margin-bottom: 5px;"/>
                    <h1 style="margin-top: 5px; margin-bottom: 5px;">Point Cloud Labeling Tool</h1>
                </div>
            </div>
        """)


        def get_reference_point():
            global selected_point
            return f"x: {selected_point['x']}, y: {selected_point['y']}, z: {selected_point['z']}"

        def refresh_reference_point():
            return get_reference_point()


        with gr.Column():
            with gr.Row():
                point_cloud_file = gr.File(label="Punktwolke hochladen (.ply, .txt, .asc, .obj)")
                assembly_parts_folder_input = gr.Textbox(label="Orderpfad, in dem die CAD Daten sich befinden angeben")
                load_button = gr.Button("Laden", interactive=True)
                print("Assembly Folder: ", assembly_parts_folder)


        with gr.Row():
            with gr.Column(scale=1, elem_id="viewer1-container"):
                gr.HTML(f"""<h3 style="margin-top: 5px; margin-bottom: 5px;">As-Built Scan</h3> """)
                viewer_1 = gr.HTML(label="Viewer 1")
                #show_button1 = gr.Button("View in O3D", interactive=False)


            with gr.Column(scale=1, elem_id="viewer2-container"):
                part_in_scan_label = gr.HTML(f"""<h3 style="margin-top: 5px; margin-bottom: 5px;">As-Built vs As-Planned Part</h3> """)
                viewer_2 = gr.HTML(label="Viewer 2")

        with gr.Column():
            with gr.Row():
                back_button = gr.Button("Zurück", interactive=True)
                forward_button = gr.Button("Vor", interactive=True)
        compare_button = gr.Button("Vergleichen", interactive=True)
        with gr.Row():
            with gr.Column(scale=1, elem_id="viewer3-container"):
                part_label = gr.HTML(f"""<h3 style="margin-top: 5px; margin-bottom: 5px;">As-Planned Part</h3> """)
                viewer_3 = gr.HTML(label="Viewer 3")
            with gr.Column(scale=1, elem_id="viewer4-container"):
                gr.HTML(f"""<h3 style="margin-top: 5px; margin-bottom: 5px;">As-Built Part vs As-Planned Part</h3> """)
                viewer_4 = gr.HTML(label="Viewer 4")
            with gr.Column(scale=1, elem_id="viewer5-container"):
                gr.HTML(f"""<h3 style="margin-top: 5px; margin-bottom: 5px;">As-Built Part</h3> """)
                viewer_5 = gr.HTML(label="Viewer 5")
                download_btn = gr.Button("Download")
                download_file = gr.File(label="Download .ply", interactive=False)
            with gr.Column(scale=1):
                post_fix = gr.Textbox(label="Dateinamenanhang pro Label Scan # postfix e.g. manuel_ma_bz7 '_partid' will be added automatically ")
                current_label = gr.Textbox(label="Label - Enter 0 (not present), 1 (partially present), 2 (completely present), or 3 (another object):")
                insert_button = gr.Button("Add Labeled Data", interactive=True)
                response_text = gr.Textbox(label="Antwort", interactive=False)

        # UPDATEs
        # Punktwolke für Viewer 1 laden
        point_cloud_file.upload(load_point_cloud, inputs=point_cloud_file, outputs=viewer_1)

        assembly_parts_folder_input.change(get_folder_dir, inputs = assembly_parts_folder_input)

        #show_button1.click(show_pcd_in_o3d_viewer1, inputs= pcd_input)
        compare_button.click(compare, outputs=[viewer_3, viewer_4, viewer_5, response_text])
        load_button.click(load_part, outputs=[viewer_2,part_label, part_in_scan_label,post_fix])
        back_button.click(load_previous_part, outputs=[viewer_2,part_label,part_in_scan_label, post_fix])
        forward_button.click(load_next_part, outputs=[viewer_2,part_label,part_in_scan_label, post_fix])
        download_btn.click(fn=export_pointcloud_for_download, outputs=download_file)

        insert_button.click(save_labeled_data_in_dict, inputs=[assembly_parts_folder_input, post_fix, current_label], outputs= response_text)

        # Update the reference point output when the data changes

    return demo

#########################################################################################################
#Flask-Server einrichten
app = Flask(__name__)

# Globale Variablen für Punktwolken
pcd_input = None
pcd_input_down = None
points_input = None
as_built_segmented = None
pcd_model = None
pcd_viewer1 = None
pcd_viewer2= None
pcd_as_built_on_surface = None
assembly_parts_folder = ""
part_file_name = ''
part_id = 0
port = 8001

# Globale Variablen für Punktwolken
selected_point = {"x": 0, "y": 0, "z": 0}


@app.route('/point', methods=['POST'])
def receive_point():
    global selected_point
    data = request.json
    selected_point = {"x": data.get('x'), "y": data.get('y'), "z": data.get('z')}
    print(f"Received point coordinates: x={selected_point['x']}, y={selected_point['y']}, z={selected_point['z']}")
    return jsonify({'status': 'success', 'x': selected_point['x'], 'y': selected_point['y'], 'z': selected_point['z']})


# Verzeichnis, in dem die HTML-Dateien liegen
HTML_DIR = os.path.join(os.getcwd(), "html_files")
Logo_DIR = os.path.join(os.getcwd(), "logo")
if not os.path.exists(HTML_DIR):
    os.makedirs(HTML_DIR)
if not os.path.exists(Logo_DIR):
    os.makedirs(Logo_DIR)



@app.route('/<path:filename>')
def serve_html(filename):
    response = send_from_directory(HTML_DIR, filename)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    return response

# Thread starten, um den Flask-Server auszuführen
def start_flask():
    global port
    app.run(port=port, debug=False)


thread = Thread(target=start_flask)
thread.start()

if __name__ == "__main__":
    demo = build_interface()
    demo.launch()
