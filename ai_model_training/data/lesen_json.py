import json
import os
import numpy as np
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
import ijson
import json
from collections import defaultdict


import orjson
import ijson

import json
import ijson
from decimal import Decimal


def combine_2_datasets2(file1, file2, output_file):
    """
    Merges two large JSON files with minimal RAM usage.
    - Handles Decimal objects properly
    - Uses streaming JSON parsing
    - Checks for key conflicts
    - Writes output incrementally
    """

    def decimal_encoder(obj):
        """Custom JSON encoder that handles Decimal objects"""
        if isinstance(obj, Decimal):
            return float(obj)  # or str(obj) if you need exact precision
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    def stream_json_kv(file_path):
        """Stream key-value pairs from a JSON file"""
        with open(file_path, 'rb') as f:
            parser = ijson.kvitems(f, '')
            for k, v in parser:
                yield k, v

    # First pass: Check for key conflicts
    print("🔍 Checking for key conflicts...")
    keys_in_file1 = set()
    for k, _ in stream_json_kv(file1):
        keys_in_file1.add(k)

    for k, _ in stream_json_kv(file2):
        if k in keys_in_file1:
            print(f"❌ Key conflict detected: '{k}' exists in both files!")
            return

    # Second pass: Merge and write output
    print("🔄 Merging files (streaming mode)...")
    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.write('{')

        first_entry = True

        # Process file1
        for k, v in stream_json_kv(file1):
            if not first_entry:
                outfile.write(',')
            json_value = json.dumps(v, default=decimal_encoder)
            outfile.write(f'"{k}":{json_value}')
            first_entry = False

        # Process file2
        for k, v in stream_json_kv(file2):
            if not first_entry:
                outfile.write(',')
            json_value = json.dumps(v, default=decimal_encoder)
            outfile.write(f'"{k}":{json_value}')
            first_entry = False

        outfile.write('}')

    print("✅ Merge completed successfully!")

def combine_2_datasets(file1, file2):
    import json
    import sys
    import ijson
    import orjson

    output_file = "dataset_obj_verification_esf_extended_part_merged_0_1_2_3_manuel.json"

    def stream_json_file(file_path):
        with open(file_path, 'rb') as f:
            parser = ijson.kvitems(f, '')  # top-level key-value pairs
            for k, v in parser:
                yield k, v

    merged = {}

    # Erstes File streamen
    for k, v in stream_json_file(file1):
        merged[k] = v

    # Zweites File streamen & auf Konflikte prüfen
    for k, v in stream_json_file(file2):
        if k in merged:
            print(f"❌ Konflikt bei Schlüssel: {k}")
            exit(1)
        merged[k] = v

    # Speichern mit orjson
    with open(output_file, 'wb') as out:
        out.write(orjson.dumps(merged, option=orjson.OPT_INDENT_2))

    print("✅ Merge abgeschlossen")


import json
from collections import Counter

def count_labels_in_jsonl(jsonl_path):
    label_counter = Counter()

    with open(jsonl_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            label = entry.get("label")
            if label is not None:
                label_counter[label] += 1

    print("📊 Anzahl pro Label:")
    for label, count in sorted(label_counter.items()):
        print(f"Label {label}: {count} Instanzen")


def clean_dataset_esf(input_path, output_path):
    valid_count = 0
    invalid_count = 0

    with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
        for idx, line in enumerate(infile):
            try:
                entry = json.loads(line)
                esf_diff = entry.get("esf_diff")

                if isinstance(esf_diff, list) and len(esf_diff) == 660:
                    outfile.write(json.dumps(entry) + "\n")
                    valid_count += 1
                else:
                    invalid_count += 1

            except json.JSONDecodeError as e:
                invalid_count += 1
                continue  # Skip malformed lines

    print(f"✅ Bereinigung abgeschlossen")
    print(f"✔️  Gültige Einträge: {valid_count}")
    print(f"❌ Entfernte ungültige Einträge: {invalid_count}")
    print(f"📄 Neue Datei: {output_path}")


import re
def search_classes(key_file_txt = "keys_files.txt"):
    import re
    from collections import defaultdict

    # === Lade alle Instanznamen ===
    with open("extrahierte_basisnamen.txt", "r") as f:
        zeilen = [z.strip() for z in f if z.strip()]

    # === Typ-2 Klassen (ModelNet40) ===
    modelnet40_klassen = {
        "airplane", "bathtub", "bed", "bench", "bookshelf", "bottle", "bowl",
        "car", "chair", "cone", "cup", "curtain", "desk", "door", "dresser",
        "flower_pot", "glass_box", "guitar", "keyboard", "lamp", "laptop",
        "mantel", "monitor", "night_stand", "person", "piano", "plant",
        "radio", "range_hood", "sink", "sofa", "stairs", "stool", "table",
        "tent", "toilet", "tv_stand", "vase", "wardrobe", "xbox"
    }

    # === Regeln ===
    def finde_klasse(instanzname):
        if "_label_2_same" in instanzname:
            return None  # ignorieren

        if "manuel" in instanzname:
            return "as_planned_LW_Campus_VAL"

        if "_label_3_other_obj" in instanzname:
            # Suche nach allen vorkommenden ModelNet40 Klassen im Namen
            gefunden = []
            for klasse in modelnet40_klassen:
                if re.search(rf"[_-]{klasse}[_-]", instanzname):
                    gefunden.append(klasse)
            return gefunden if gefunden else ["label_3_other_obj"]

        if re.match(r"^as_planned_c\d_", instanzname):
            return "as_planned_concrete"

        if any(instanzname.startswith(prefix) for prefix in [
            "as_planned_IFC_SKW_Modell_07052019",
            "as_planned_IoC_01_demonstrator",
            "as_planned_LW_Campus_VAL",
            "steelbeam_as_planned",
            "steelcon_assembly"
        ]):
            return instanzname.split("_", max(instanzname.count("_") - 1))[0]

        for klasse in modelnet40_klassen:
            if instanzname.startswith(klasse + "_"):
                return klasse

        return "klasse_unbekannt"

    # === Erstelle das Dictionary ===
    class_instance_dict = defaultdict(list)

    for instanz in zeilen:
        klassen = finde_klasse(instanz)
        if not klassen:
            continue
        if isinstance(klassen, list):
            for k in klassen:
                class_instance_dict[k].append(instanz)
        else:
            class_instance_dict[klassen].append(instanz)

    # === Speichern als Textdatei ===
    with open("instanz_klassenzuordnung.txt", "w") as f:
        for cls, instanzen in sorted(class_instance_dict.items()):
            f.write(f"{cls}:\n")
            for inst in instanzen:
                f.write(f"  - {inst}\n")
            f.write("\n")

# Beispiel:
#count_labels_in_jsonl("dataset_obj_verification_esf_extended_binary_oneshot.jsonl")


# check wrong data

import json

# Lade beide Dateien
with open("verf_esf_dataset_3_instances_merged.json", 'r') as f1:
    data1 = json.load(f1)

with open("verf_normal_xray_hist_dataset_3_instances_merged.json", 'r') as f2:
    data2 = json.load(f2)

# Durchlaufe gemeinsam
for items in data1:
    if items not in data2:
        print(f"⚠️ {items} fehlt in zweiter Datei")
        continue
    # beide haben gleiche Struktur
    for each in data1[items]:
        if each not in data2[items]:
            print(f"⚠️ {each} fehlt in {items} in zweiter Datei")
            continue
        for perc in data1[items][each]:
            if perc not in data2[items][each]:
                print(f"⚠️ {perc} fehlt in {items}->{each} in zweiter Datei")
                continue
            len1 = len(list(data1[items][each][perc]))
            len2 = len(list(data2[items][each][perc]))
            print(f"{items} : {each} : {perc} : {len1} vs {len2}")



# json_path = "verf_esf_dataset_2.json"
#
# with open(json_path, 'r') as file:
#     data = json.load(file)
#     for i, items in enumerate(data):
#         print(items, data[items].keys())
#         for each in data[items].keys():
#             percs = data[items][each].keys()
#             for perc in percs:
#                 print(items, ":", each, ":", perc, ":", len(list(data[items][each][perc])) )
#
# json_path = "verf_normal_xray_hist_dataset.json"
#
# with open(json_path, 'r') as file:
#     data = json.load(file)
#     for i, items in enumerate(data):
#         print(items, data[items].keys())
#         for each in data[items].keys():
#             percs = data[items][each].keys()
#             for perc in percs:
#                 print(items, ":", each, ":", perc, ":", len(list(data[items][each][perc])) )

#search_classes()

# suchmuster = "_1kpts_2"
# extrahierte_namen = []
#
# with open("keys_files.txt", "r") as f:
#     for zeile in f:
#         key = zeile.strip()
#         if suchmuster in key:
#             basis = key.split(suchmuster)[0]
#             extrahierte_namen.append(basis)
#
# # In Datei schreiben (optional)
# with open("extrahierte_basisnamen.txt", "w") as f:
#     for name in extrahierte_namen:
#         f.write(name + "\n")
#
# print(f"{len(extrahierte_namen)} Namen extrahiert.")
#
# with open("keys_files.txt", "r") as f:
#     keys = [line.strip() for line in f]
#     print(keys[500:1000])