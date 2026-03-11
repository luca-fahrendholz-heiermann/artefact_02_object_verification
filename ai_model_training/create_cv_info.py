import json
import random
from sklearn.model_selection import KFold


# Dateien einlesen
def load_keys_from_files(file1, file2):
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        data1 = json.load(f1)
        data2 = json.load(f2)
    keys = list(data1.keys()) + list(data2.keys())
    return keys


# Cross-Validation-Daten erstellen
def create_cross_validation_splits_with_keys(keys, num_splits=5):
    random.shuffle(keys)
    kf = KFold(n_splits=num_splits, shuffle=True, random_state=42)
    splits = {}
    for fold, (train_idx, test_idx) in enumerate(kf.split(keys)):
        train_keys = [keys[i] for i in train_idx]
        test_keys = [keys[i] for i in test_idx]
        val_split = int(0.2 * len(train_keys))  # 20% für Validierung
        val_keys = train_keys[:val_split]
        train_keys = train_keys[val_split:]
        splits[fold] = {
            "train_data": [f"pc_id={key}" for key in train_keys],
            "val_data": [f"pc_id={key}" for key in val_keys],
            "test_data": [f"pc_id={key}" for key in test_keys],
        }
    return splits


import jsonu
import random
import numpy as np
from collections import defaultdict
from sklearn.model_selection import KFold


# Dateien einlesen und ESF-Vektoren laden
def load_data_with_esf_check(file_path):
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
def create_cross_validation_splits_with_balanced_val(data, labels, num_splits=5, val_ratio=0.2):
    indices = list(range(len(data)))
    random.shuffle(indices)

    # Indizes nach Label-Klassen gruppieren
    label_to_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        label_to_indices[label].append(idx)

    kf = KFold(n_splits=num_splits, shuffle=True, random_state=42)
    splits = {}

    for fold, (train_idx, test_idx) in enumerate(kf.split(indices)):
        train_idx = list(train_idx)
        test_idx = list(test_idx)

        # Validierungsdaten proportional zur Label-Verteilung erstellen
        val_indices = []
        remaining_train_indices = train_idx.copy()

        # Berechne die Anzahl der Validierungsdaten für jedes Label
        for label, label_indices in label_to_indices.items():
            available_indices = [idx for idx in train_idx if idx in label_indices]
            val_count = int(val_ratio * len(train_idx) * (len(available_indices) / len(train_idx)))
            selected_val_indices = random.sample(available_indices, min(val_count, len(available_indices)))
            val_indices.extend(selected_val_indices)

            # Entferne ausgewählte Validierungsdaten aus den Trainingsdaten
            remaining_train_indices = [idx for idx in remaining_train_indices if idx not in selected_val_indices]

        splits[fold] = {
            "train_indices": remaining_train_indices,
            "val_indices": val_indices,
            "test_indices": test_idx,
        }

    return splits






# Ergebnis speichern
def save_to_json(data, output_file):
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

# Hauptfunktion
def main():
    file_path = 'dataset_obj_verification_esf_extended_part_merged_1_2_3_manuel.json'
    output_file = 'cv_6_info_balanced.json'

    # Daten laden
    esf_data, labels = load_data_with_esf_check(file_path)

    # Cross-Validation-Splits erstellen
    splits = create_cross_validation_splits_with_balanced_val(esf_data, labels, num_splits=6)

    # Ergebnisse speichern
    save_to_json(splits, output_file)

    # Anzahl der Datensätze ausgeben
    print(f"Gesamtzahl der Daten: {len(esf_data)}")
    for fold, split in splits.items():
        print(f"Fold {fold}:")
        print(
            f"  Training: {len(split['train_indices'])} | Validation: {len(split['val_indices'])} | Test: {len(split['test_indices'])}")

    print(f"Cross-Validation-Splits mit proportionalen Validierungsdaten wurden in {output_file} gespeichert.")

# Hauptfunktion
def main2():
    file1 = 'dataset_obj_verification_esf_extended_part_merged_1_2_3_manuel.json'
    file2 = 'dataset_obj_verification_esf_extended_part2.json'
    output_file = 'cross_validation_splits.json'

    #keys = load_keys_from_files(file1, file2)
    with open(file1, 'r') as f1:
        data = json.load(f1)
    keys = list(data.keys())
    splits = create_cross_validation_splits_with_keys(keys, num_splits=6)
    save_to_json(splits, output_file)
    print(f"Cross-Validation-Splits wurden in {output_file} gespeichert.")


if __name__ == "__main__":
    main()