import json


# Dateien einlesen und zusammenführen
def merge_json_files(file1, file2, output_file):
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        data1 = json.load(f1)
        data2 = json.load(f2)

    # Zusammenführen der Dictionaries
    merged_data = {**data1, **data2}

    # In eine neue Datei schreiben
    with open(output_file, 'w') as f_out:
        json.dump(merged_data, f_out, indent=4)

    print(f"Die Dateien wurden erfolgreich zusammengeführt und in {output_file} gespeichert.")



def merge_json_files2(file1, file2, output_file):
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        data1 = json.load(f1)
        data2 = json.load(f2)

    merged_data = {}

    for key, value in data1.items():
        merged_data[key] = value

    for key, value in data2.items():
        if key in merged_data:
            counter = 1
            new_key = f"{key}_{counter}"
            while new_key in merged_data:
                counter += 1
                new_key = f"{key}_{counter}"
            merged_data[new_key] = value
        else:
            merged_data[key] = value

    with open(output_file, 'w') as f_out:
        json.dump(merged_data, f_out, indent=4)

    print(f"Die Dateien wurden erfolgreich zusammengeführt und in {output_file} gespeichert.")

    print(f"Die Dateien wurden erfolgreich zusammengeführt und in {output_file} gespeichert.")


# Hauptfunktion
def main():
    file1 = 'dataset_obj_verification_esf_extended_part_merged_1_2_3_manuel_extend.json'
    file2 = 'dataset_obj_verification_esf_extended_part_label0_2.json'
    output_file = 'dataset_obj_verification_esf_extended_part_merged_1_2_3_manuel.json'

    merge_json_files2(file1, file2, output_file)


if __name__ == "__main__":
    main()