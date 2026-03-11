import json
import random
import os
from sklearn.model_selection import KFold


def generate_esf_diff_pair_dataset(esf_data_file, output_file="cv6_info.json"):
    # 📥 1. Lade ESF-Daten
    with open(esf_data_file, "r") as f:
        esf_dict = json.load(f)

    # 🔎 2. Alle Instanzen sammeln
    all_instances = [(cls, inst_id) for cls in esf_dict for inst_id in esf_dict[cls]]

    # ⚙️ 3. Cross-Validation vorbereiten
    kf = KFold(n_splits=6, shuffle=True, random_state=42)
    #90, 80, 70, 60, 58, 45, 35
    keys_label_1 = ['30','35', '40','45','50', '55','58', '60', '65','70', '75', '80', '85','90', '95', '100']
    keys_label_0 = ['10']

    # Hier sammeln wir alle Folds in einem dict
    all_folds_data = {}

    # 🔁 4. Bearbeite alle Folds
    for fold_idx, (trainval_idx, test_idx) in enumerate(kf.split(all_instances)):
        print(f"\n🧩 Starte Fold {fold_idx}")
        trainval = [all_instances[i] for i in trainval_idx]
        test = [all_instances[i] for i in test_idx]
        val_size = int(0.2 * len(trainval))
        random.shuffle(trainval)
        val = trainval[:val_size]
        train = trainval[val_size:]

        data = {"train": [], "val": [], "test": []}

        def generate_label0_1(instances, split_key):
            for cls, inst_id in instances:
                instance = esf_dict[cls][inst_id]
                if "100" not in instance:
                    continue
                esf_100 = instance["100"]
                if not esf_100 or len(esf_100[0]) != 640:
                    continue
                ref_id = f"{cls}_{inst_id}_100_0"

                # Label 1
                for key in keys_label_1:
                    if key not in instance:
                        continue
                    for j, scan_vec in enumerate(instance[key]):
                        if key == "100" and j == 0:
                            continue
                        if len(scan_vec) == 640:
                            scan_id = f"{cls}_{inst_id}_{key}_{j}"
                            data[split_key].append({
                                "esf_ref": ref_id,
                                "esf_scan": scan_id,
                                "label": 1
                            })

                # Label 0
                for key in keys_label_0:
                    if key not in instance:
                        continue
                    for j, scan_vec in enumerate(instance[key]):
                        if len(scan_vec) == 640:
                            scan_id = f"{cls}_{inst_id}_{key}_{j}"
                            data[split_key].append({
                                "esf_ref": ref_id,
                                "esf_scan": scan_id,
                                "label": 0
                            })

        def generate_label2(instances, split_key):
            # Welche Prozent-Keys für "anderes Objekt" verwendet werden sollen
            neg_keys = ['50', '55', '58', '60', '65', '70', '75', '80', '85', '90', '95', '100']

            # Pro Key limitieren, damit der Datensatz nicht explodiert
            # (anpassen, wenn du mehr/weniger willst)
            max_per_key = {
                '100': 2,
                '50': 3,
                # alle anderen Keys → default_max
            }
            default_max = 3

            # Instanzen je Klasse sammeln
            class_to_instances = {}
            for cls, inst_id in instances:
                class_to_instances.setdefault(cls, []).append(inst_id)

            for ref_class, ref_list in class_to_instances.items():
                if not ref_list:
                    continue

                # Referenzen sampeln (max 5 pro Klasse wie zuvor)
                ref_inst_ids = random.sample(ref_list, min(5, len(ref_list)))
                for ref_inst_id in ref_inst_ids:
                    try:
                        ref_vec = esf_dict[ref_class][ref_inst_id]["100"][0]
                    except Exception:
                        continue
                    if len(ref_vec) != 640:
                        continue

                    ref_id = f"{ref_class}_{ref_inst_id}_100_0"

                    # Vergleichs-Kandidaten: alle anderen Instanzen (auch aus anderer Klasse)
                    for target_class, target_list in class_to_instances.items():
                        candidates = [i for i in target_list
                                      if not (target_class == ref_class and i == ref_inst_id)]
                        if not candidates:
                            continue

                        comp_inst_ids = random.sample(candidates, min(5, len(candidates)))
                        for comp_inst_id in comp_inst_ids:
                            comp_data = esf_dict[target_class].get(comp_inst_id, {})
                            # über alle gewünschten Prozent-Keys iterieren
                            for key in neg_keys:
                                esf_list = comp_data.get(key, [])
                                # nur gültige 640er-Vektoren nehmen
                                valid_idx = [i for i, v in enumerate(esf_list) if isinstance(v, list) and len(v) == 640]
                                if not valid_idx:
                                    continue
                                kmax = max_per_key.get(key, default_max)
                                take = min(kmax, len(valid_idx))
                                for i in random.sample(valid_idx, take):
                                    data[split_key].append({
                                        "esf_ref": ref_id,
                                        "esf_scan": f"{target_class}_{comp_inst_id}_{key}_{i}",
                                        "label": 2
                                    })

        # 🔧 Vergleiche generieren
        generate_label0_1(train, "train")
        generate_label2(train, "train")
        generate_label0_1(val, "val")
        generate_label2(val, "val")
        generate_label0_1(test, "test")
        generate_label2(test, "test")

        # 📦 Füge Fold-Daten zum großen Dict hinzu
        all_folds_data[f"fold{fold_idx}"] = data

        print(f"✅ Fold {fold_idx} generiert → Train: {len(data['train'])}, Val: {len(data['val'])}, Test: {len(data['test'])}")

    # 💾 Alle Folds in eine Datei speichern
    with open(output_file, "w") as f:
        json.dump(all_folds_data, f)

    print(f"\n✅ Alle Folds in {output_file} gespeichert.")

    # === 2. Analyse der Splits in allen Folds ===
    for i in range(6):
        fold_path = f"cv_fold_{i}/data.json"
        if not os.path.exists(fold_path):
            print(f"⚠️  Datei fehlt: {fold_path}")
            continue

        with open(fold_path, "r") as f:
            data = json.load(f)

        print(f"🧩 Fold {i}")
        print(f"📁 Gespeichert: {fold_path}")

        for split in ["train", "val", "test"]:
            total = len(data[split])
            count_0 = sum(1 for s in data[split] if s["label"] == 0)
            count_1 = sum(1 for s in data[split] if s["label"] == 1)
            count_2 = sum(1 for s in data[split] if s["label"] == 2)

            # Extrahiere Klassen aus esf_ref/scan
            classes = set()
            for s in data[split]:
                for k in ["esf_ref", "esf_scan"]:
                    try:
                        class_key = s[k]
                        classes.add(class_key)
                    except:
                        continue

            print(f"  - {split}: {total} Samples → Label 0: {count_0}, Label 1: {count_1}, Label 2: {count_2}")
            print(f"    Klassen in {split}: {sorted(list(classes))}")
        print()

    print("\n✅ Alle Folds erfolgreich strukturiert generiert.")


if __name__ == "__main__":
    file = "verf_esf_dataset_3_instances_merged.json"

    import json
    import os
    from collections import defaultdict, Counter

    # === 1. Lade und analysiere ESF-Daten ===
    with open(file, "r") as f:
        esf_dict = json.load(f)

    total_instances = 0
    valid_esf_count = 0
    invalid_esf_count = 0
    class_instance_counts = defaultdict(int)
    instance_keys = []
    inst_keys_with_percents = {}
    instanz_vektor_anzahl = {}

    for cls, inst_dict in esf_dict.items():
        for inst_id, inst_data in inst_dict.items():
            total_instances += 1
            class_instance_counts[cls] += 1
            inst_key = f"{cls}_{inst_id}"
            instance_keys.append(inst_key)
            perc_keys = list(inst_data.keys())
            inst_keys_with_percents[inst_key] = perc_keys

            # === NEU: Zähle gültige ESF-Vektoren pro Prozent und gesamt
            prozent_dict = {}
            total_vectors = 0

            for perc_key, esf_list in inst_data.items():
                valid_count = sum(1 for vec in esf_list if isinstance(vec, list) and len(vec) == 640)
                prozent_dict[perc_key] = valid_count
                total_vectors += valid_count

                # Zähle global
                valid_esf_count += valid_count
                invalid_esf_count += len(esf_list) - valid_count

            instanz_vektor_anzahl[inst_key] = {
                "per_prozent": prozent_dict,
                "gesamt": total_vectors
            }

    # === 📊 Ausgaben ===
    print(f"\n🔢 Gesamtzahl der Instanzen: {total_instances}")
    print(f"✅ Gültige ESF-Vektoren (len == 640): {valid_esf_count}")
    print(f"❌ Ungültige / leere ESF-Vektoren: {invalid_esf_count}\n")

    print("📊 Instanzen pro Klasse:")
    for cls, count in sorted(class_instance_counts.items()):
        print(f"  - {cls}: {count}")

    print(f"\n🗝️ Instanz-Keys (class_instanz): {len(instance_keys)} total")
    print(instance_keys[:10], "...")

    print("\n🔑 Prozent-Keys pro Instanz (Auszug):")
    for key in list(inst_keys_with_percents.keys())[:10]:
        print(f"  - {key}: {inst_keys_with_percents[key]}")

    # === Analyse: wie viele Prozent-Keys pro Instanz vorhanden? ===
    count_perc_keys = Counter(len(v) for v in inst_keys_with_percents.values())

    print("\n📈 Verteilung Anzahl Prozent-Keys pro Instanz:")
    for n_keys, count in sorted(count_perc_keys.items()):
        print(f"  - {n_keys} Prozent-Keys: {count} Instanzen")

    # === NEU: Ausgabe ESF-Vektor-Anzahl pro Instanz ===
    print("\n🧮 Gültige ESF-Vektoren pro Instanz (Auszug):")
    for key in list(instanz_vektor_anzahl.keys())[:10]:
        data = instanz_vektor_anzahl[key]
        print(f"  - {key}: Gesamt = {data['gesamt']}, Verteilung = {data['per_prozent']}")

    generate_esf_diff_pair_dataset(file)