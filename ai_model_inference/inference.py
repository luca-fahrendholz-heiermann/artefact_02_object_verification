import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import os
import sys
import open3d as o3d
import json
import random
import json
import open3d as o3d
import numpy as np
import os
import time
import copy
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.neighbors import KDTree
import point_cloud_descriptors as pc_desc
from scipy.spatial.distance import cosine
from scipy.stats import wasserstein_distance
import threading
import time
import laspy as lp



dir_model = os.path.join(os.getcwd(), "model")
dir_input = os.path.join(os.getcwd(), "input")
dir_input_model = os.path.join(dir_input, "model")
dir_input_scan = os.path.join(dir_input, "scan")
dir_model_checkpoint = os.path.join(os.getcwd(), "checkpoint")
sys.path.append(dir_model)
sys.path.append(dir_input_scan)
sys.path.append(dir_input_model)
sys.path.append(dir_input)

from ov_ai_model import OV_RESNET

# read pointcloud in any format:
def read_pcd_in_any_format(input_data):
    has_feats = False
    # Punktwolke einlesen
    if isinstance(input_data, np.ndarray):
        print("Der Parameter ist ein numpy array.")
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(input_data[:, :3])


    elif isinstance(input_data, str):
        print("Der Parameter ist ein string.")

        dir_pcd_path = input_data

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
                rgb_list = [[r, g, b] for r, g, b in
                            zip(np.zeros(num_points), np.zeros(num_points), np.zeros(num_points))]
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

def normalize_to_minus_one_and_one_v2(descriptor):
    max_abs_val = np.max(np.abs(descriptor))

    # Avoid division by zero if max_abs_val is zero
    if max_abs_val == 0:
        return np.zeros_like(descriptor)  # If all values are zero, return zero vector

    # Normalize to range [-1, 1] while keeping zero centered
    normalized_descriptor = descriptor / max_abs_val
    return normalized_descriptor

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


def preprocess_data(model_pcd, scan_pcd):
    # Create ESF Descriptors for both Point Clouds (Model and Scan)
    esf_model = pc_desc.compute_esf_descriptor(pcd_o3d=model_pcd)
    esf_model = normalize_to_minus_one_and_one_v2(esf_model)
    plot_histogram(esf_model, label="model")

    esf_scan = pc_desc.compute_esf_descriptor(pcd_o3d=scan_pcd)
    esf_scan = normalize_to_minus_one_and_one_v2(esf_scan)
    plot_histogram(esf_scan, label="scan")

    # Create ESF Descriptors Difference
    esf_diff = np.array(esf_model) - np.array(esf_scan)
    esf_scan_diff_norm = normalize_to_minus_one_and_one_v2(esf_diff)
    plot_histogram(esf_scan_diff_norm, label="diff")

    # Create EMD Values
    emd_values = compute_emd_for_esf(esf_model, esf_scan)
    # Create COSINE Similarity
    cosine_similarity_values = calculate_cosine_similarity(esf_model, esf_scan)

    # Merge all Features to one Vector of 660 Values
    esf_diff_extended_norm = np.concatenate((esf_scan_diff_norm, emd_values, cosine_similarity_values))
    esf_diff_extended_norm = np.round(esf_diff_extended_norm, decimals=4)
    plot_histogram(esf_diff_extended_norm, label="diff_extended")

    return esf_diff_extended_norm



def ov_inference(input_model, input_scan):
    # Hyperparameter
    # Check if GPU is available
    print(torch.version.cuda)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Current device number is: {torch.cuda.current_device}")
    print(f"GPU name is {torch.cuda.get_device_name(torch.cuda.current_device)}")

    # 🔧 Hyperparameter

    setup = {
        "input_size": 660,
        'hidden_layers': [512, 512, 256, 256, 128, 128], # [512, 256, 128, 64],  # Beispiel-Konfiguration # default [512, 256, 128, 64], res : [512, 512, 256, 256, 128, 128]
        "activation_f": "tanh",  # "leaky_relu", "relu" ,"tanh"
        "batch_size": 128,
        "batch_size_val": 512,
        "model": "res",  # res, mlp,transformer
        "dropout_rate": 0.1,
        "output_size": 4
    }

    model_path = os.path.join(dir_model_checkpoint, os.listdir(dir_model_checkpoint)[0])

    # Load Data
    pcd_model = read_pcd_in_any_format(input_model)
    pcd_scan = read_pcd_in_any_format(input_scan)

    # Generate Input format for AI Inference
    esf_diff_extended_norm = preprocess_data(pcd_model, pcd_scan) # Input Vector of 660 Values
    input_tensor = torch.tensor(esf_diff_extended_norm, dtype=torch.float32).unsqueeze(0).to(device)

    # Load Trained AI Model for Object Verification
    model = OV_RESNET(input_size=setup["input_size"], hidden_layers=setup['hidden_layers'], dropout_rate=setup['dropout_rate'], activation_function=setup["activation_f"], output_size = setup["output_size"])
    model.load_state_dict(torch.load(model_path, map_location=device))
    num_outputs = model.output_layer.out_features
    print(f"Number of outputs: {num_outputs}")
    model.to(device)
    model.eval()

    # 🔮 Inferenz
    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1).item()
        confidence = torch.softmax(output, dim=1).squeeze().cpu().numpy()

    # 📊 Ergebnis
    print(f"[RESULT] Prediction: {prediction} | Confidence: {confidence}")

    classification_labels = {0:"Objekt nicht vorhanden", 1: "Objekt vorhanden", 2: "Anderes Objekt", 3: "keine Klasse"}

    label = classification_labels[prediction]
    return prediction, confidence, label

if __name__ == "__main__":
    start = time.time()
    input_model = os.path.join(dir_input_model, os.listdir(dir_input_model)[0])
    input_scan = os.path.join(dir_input_scan, os.listdir(dir_input_scan)[0])
    prediction, confidence, label = ov_inference(input_model, input_scan)
    print("Prediction Status: ", prediction)
    print("Confidence: ", confidence)
    print(label)
    print("Inference Zeit: ", time.time() -start, " Sekunden")