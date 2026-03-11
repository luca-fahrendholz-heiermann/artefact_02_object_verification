                                                        import json
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from collections import Counter
import random
from sklearn.model_selection import StratifiedKFold
from collections import Counter
from itertools import chain
from sklearn.model_selection import train_test_split


# Dateien einlesen und ESF-Vektoren laden
def load_data_with_esf_check(file_path):
    """
    Lädt die Daten und filtert ESF-Vektoren mit der richtigen Form (660).
    Gibt ESF-Daten, Labels und die zugehörigen Indizes zurück.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)

    esf_data = []
    labels = []
    for values in data.values():
        esf_diff = np.array(values['esf_diff'])
        if esf_diff.shape[0] == 660:
            esf_data.append(esf_diff)
            labels.append(int(values['label']))

    return esf_data, labels


# Cross-Validation-Splits mit proportionaler Verteilung der Labels erstellen
def create_cross_validation_splits_with_indices_stratified(data, labels, num_splits=5, ignore_list  = None):
    """
    Erstellt Cross-Validation-Splits basierend auf Indizes und sorgt für native Python-Typen.
    """
    indices = list(range(len(data)))
    print(len(indices))
    print(len(labels))

    if ignore_list != None:
        indices = np.array(indices)
        # Create a mask to exclude ignore_list
        mask = ~np.isin(indices, ignore_list)
        # Apply the mask to get valid indices
        valid_indices = indices[mask]
        valid_labels = np.array(labels)[valid_indices]
        print(len(valid_indices))
        print(len(valid_labels))

        indices = list(valid_indices)
        labels = list(valid_labels)
    random.shuffle(indices)

    kf =  StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=42)
    splits = {}
    for fold, (train_idx, test_idx) in enumerate(kf.split(indices, labels)):
        # Teile die Trainingsdaten in Trainings- und Validierungssatz
        train_labels = np.array(labels)[train_idx]
        skf_val = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        # Iteriere über die Folds und hole die entsprechenden Indizes
        for train_split_idx, val_split_idx in skf_val.split(train_idx, train_labels):
            if fold == fold:  # Nur den ersten Split verwenden
                train_indices = train_idx[train_split_idx]
                val_indices = train_idx[val_split_idx]
                break

        # Konvertiere NumPy-Typen in native Python-Typen
        splits[fold] = {
            "train_indices": [int(idx) for idx in train_indices],
            "val_indices": [int(idx) for idx in val_indices],
            "test_indices": [int(idx) for idx in test_idx],
        }

    return splits


def train_final_model_stratified(data, labels, test_size=0.1, random_state=42):
    """
    Train a final model with stratified 90-10 train-validation split.

    Args:
        data: List of input data
        labels: List of corresponding labels
        test_size: Proportion of data to use for validation (default: 0.1)
        random_state: Random seed for reproducibility

    Returns:
        Dictionary containing:
        - train_indices: Indices for training data
        - val_indices: Indices for validation data
        - model: Trained model (if applicable)
    """
    # Convert to numpy arrays if they aren't already
    labels = np.array(labels)
    indices = np.arange(len(data))

    # Perform stratified split
    train_idx, val_idx = train_test_split(
        indices,
        test_size=test_size,
        stratify=labels,
        random_state=random_state
    )

    # Convert indices to Python int (from numpy int)
    result = {
        "train_indices": [int(idx) for idx in train_idx],
        "val_indices": [int(idx) for idx in val_idx]
    }

    return result


# Ergebnis speichern
def save_to_json(data, output_file):
    """
    Speichert die Splits in JSON-Format.
    """
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)


# Hauptfunktion
def main():
    file_path = 'dataset_obj_verification_esf_extended_part_merged_0_1_2_3_manuel.json'
    #data_ignore = "wrong_labeled_data_fold_0_1_2_3_4_5.json"

    #with open(data_ignore, 'r') as file:
    #    dict_data_ignore = json.load(file)

    # Combine all lists into one
    #data_ignore = list(chain.from_iterable(dict_data_ignore.values()))

    #print(len(data_ignore))



    output_file = 'cv_6_info_indices_3_classes_new_final.json'

    # Daten laden
    esf_data, labels = load_data_with_esf_check(file_path)

    # Count occurrences of each unique label
    unique_labels, counts = np.unique(labels, return_counts=True)

    # Display the results
    for label, count in zip(unique_labels, counts):
        print(f"Label {label}: {count} occurrences")

    # Labels in NumPy-Array ändern
    labels = np.array(labels)
    labels[labels == 2] = 1
    labels[labels == 3] = 2

    # Count occurrences of each unique label
    unique_labels, counts = np.unique(labels, return_counts=True)

    # Display the results
    for label, count in zip(unique_labels, counts):
        print(f"Label {label}: {count} occurrences")

    # final model:

    # Create final stratified split
    split = train_final_model_stratified(esf_data, labels)

    # Save results
    save_to_json(split, output_file)
    print(f"Final model split indices were saved in {output_file}.")

    # # Cross-Validation-Splits erstellen
    # #splits = create_cross_validation_splits_with_indices_stratified(esf_data, labels, num_splits=6, ignore_list=data_ignore)
    # splits = create_cross_validation_splits_with_indices_stratified(esf_data, labels, num_splits=6)
    #
    # # Ergebnisse speichern
    # save_to_json(splits, output_file)
    # print(f"Cross-Validation-Splits mit Indizes wurden in {output_file} gespeichert.")

def visualize_input_cnn_matrix(data, aspect = "equal"):
    # Titel für jeden Kanal
    titles = ["Histogramme", "EMD-Werte", "Cosine Similarity"]

    # Plot
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))  # 1 Zeile, 3 Spalten

    for i in range(3):
        im = axs[i].imshow(data[i], cmap='viridis', aspect=aspect, vmin=0, vmax=1) # aspect = "auto"
        axs[i].set_title(titles[i])
        axs[i].set_xlabel("Bins (64)")
        axs[i].set_ylabel("Position (10)")
        fig.colorbar(im, ax=axs[i])

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    main()

    import numpy as np
    import matplotlib.pyplot as plt

    # Kanal 0: Zufällige Histogramme (10 Positionen × 64 Bins)
    histograms = np.random.rand(10, 64)

    # Kanal 1: 10 unterschiedliche EMD-Werte, je auf eine Zeile gestreckt
    emd_values = np.linspace(0.1, 0.9, 10)  # Beispielwerte
    emd_channel = np.array([np.full(64, val) for val in emd_values])

    # Kanal 2: 10 unterschiedliche Cosine-Werte, je auf eine Zeile gestreckt
    cos_values = np.linspace(0.6, 1.0, 10)  # Beispielwerte
    cos_channel = np.array([np.full(64, val) for val in cos_values])

    # Stapeln zum 3×10×64-Tensor
    data = np.stack([histograms, emd_channel, cos_channel])

    visualize_input_cnn_matrix(data)


