import json
from collections import Counter


# Funktion, um Labels aus einer verschachtelten Struktur zu zählen
def count_labels(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Labels aus der verschachtelten Struktur extrahieren
    labels = [value["label"] for value in data.values() if "label" in value]

    # Häufigkeit der Labels berechnen
    label_counts = Counter(labels)

    print("Label-Häufigkeiten:")
    for label, count in label_counts.items():
        print(f"{label}: {count}")

    return label_counts


def ensure_keys_are_strings(input_file, output_file):
    # JSON laden
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Überprüfen und Keys in Strings umwandeln
    corrected_data = {str(key): value for key, value in data.items()}

    # Bereinigte Daten speichern
    with open(output_file, 'w') as f:
        json.dump(corrected_data, f, indent=4)

    print(f"Alle Keys wurden in Strings umgewandelt und in {output_file} gespeichert.")


# Hauptfunktion
def main():
    input_file = 'dataset_obj_verification_esf_extended_part_merged_1_2_3_manuel.json'
    output_file = 'dataset_obj_verification_esf_extended_part_merged_keys_as_strings.json'
    #ensure_keys_are_strings(input_file, output_file)

    label_counts = count_labels(output_file)



if __name__ == "__main__":
    main()
