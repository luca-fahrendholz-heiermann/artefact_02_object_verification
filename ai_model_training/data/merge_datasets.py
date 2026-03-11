import json
import orjson
from collections import defaultdict


def merge_with_filename_and_vectors(d1, d2):
    result = {}
    stats = defaultdict(lambda: defaultdict(int))  # Klasse → Instanz → Neue Prozentwerte

    for klasse in set(d1) | set(d2):
        result[klasse] = {}
        instanzen1 = d1.get(klasse, {})
        instanzen2 = d2.get(klasse, {})

        for instanz in set(instanzen1) | set(instanzen2):
            result[klasse][instanz] = {}
            p1 = instanzen1.get(instanz, {})
            p2 = instanzen2.get(instanz, {})

            # === Filename übernehmen ===
            filename1 = p1.get("filename")
            filename2 = p2.get("filename")

            if filename1 and filename2 and filename1 != filename2:
                raise ValueError(f"Filename-Konflikt bei {klasse} → {instanz}: {filename1} ≠ {filename2}")

            result[klasse][instanz]["filename"] = filename1 or filename2

            # === Prozentwerte mergen (d1 bevorzugt) ===
            keys_p1 = {k for k in p1 if k != "filename"}
            keys_p2 = {k for k in p2 if k != "filename"}

            for prozent in keys_p1 | keys_p2:
                if prozent in p1:
                    result[klasse][instanz][prozent] = p1[prozent]
                else:
                    result[klasse][instanz][prozent] = p2[prozent]
                    stats[klasse][instanz] += 1  # Zähler erhöhen

    return result, stats


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_orjson(obj, path):
    with open(path, "wb") as f:
        f.write(orjson.dumps(obj))


def print_stats(stats):
    print("\n📊 Neue Prozentwerte aus d2 übernommen:\n")
    total_new = 0
    for klasse in stats:
        for instanz in stats[klasse]:
            count = stats[klasse][instanz]
            total_new += count
            print(f"- {klasse} / Instanz {instanz}: {count} neue Prozentwerte")
    print(f"\n🧮 Gesamt: {total_new} neue Prozentwerte ergänzt.\n")


if __name__ == "__main__":
    # === Pfade ===
    dataset1_path = "verf_esf_dataset_2_instances_merged.json"
    dataset2_path = "verf_esf_dataset_3_newclasses.json"
    merged_output1 = "verf_esf_dataset_3_instances_merged.json"

    dataset3_path = "verf_normal_xray_hist_dataset_2_instances_merged.json"
    dataset4_path = "verf_normal_xray_hist_dataset3_newclasses.json"
    merged_output2 = "verf_normal_xray_hist_dataset_3_instances_merged.json"

    # === Erstes Datenset ===
    d1 = load_json(dataset1_path)
    d2 = load_json(dataset2_path)

    merged1, stats1 = merge_with_filename_and_vectors(d1, d2)
    save_orjson(merged1, merged_output1)
    print_stats(stats1)

    # === Zweites Datenset ===
    d3 = load_json(dataset3_path)
    d4 = load_json(dataset4_path)

    merged2, stats2 = merge_with_filename_and_vectors(d3, d4)
    save_orjson(merged2, merged_output2)
    print_stats(stats2)

    print("✅ Merge abgeschlossen.")