import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, SubsetRandomSampler
from sklearn.model_selection import train_test_split
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cosine
from scipy.stats import wasserstein_distance
import copy
import orjson
import random
from torch.utils.data import Subset, WeightedRandomSampler, SubsetRandomSampler, DataLoader


class ESFRefPairDatasetChannels(Dataset):
    def __init__(self, esf_data_path, cv_info_path, fold, use_metrics=False):
        self.use_metrics = use_metrics
        with open(esf_data_path, "rb") as f:
            self.esf_data = orjson.loads(f.read())

        with open(cv_info_path, "r") as f:
            cv_info = json.load(f)
        fold_info = cv_info[fold]

        # 🔥 Train/Val/Test Paare holen
        self.train_pairs = fold_info.get("train", [])
        self.val_pairs = fold_info.get("val", [])
        self.test_pairs = fold_info.get("test", [])

        # 🔥 Indizes anlegen
        self.train_indices = np.arange(0, len(self.train_pairs))
        self.val_indices = np.arange(len(self.train_pairs),
                                     len(self.train_pairs) + len(self.val_pairs))
        self.test_indices = np.arange(len(self.train_pairs) + len(self.val_pairs),
                                      len(self.train_pairs) + len(self.val_pairs) + len(self.test_pairs))

        # 🔥 Alle Paare zusammenführen
        self.all_pairs = self.train_pairs + self.val_pairs + self.test_pairs
        print(f"✅ Fold {fold}: {len(self.train_pairs)} train, {len(self.val_pairs)} val, {len(self.test_pairs)} test")

        # 🔥 Preprocessing
        self.preprocessed = []
        print("⏳ Preprocessing aller Paare ...")
        for pair in self.all_pairs:
            cls_r, inst_r, perc_r, idx_r = self._parse_esf_key(pair["esf_ref"])
            cls_s, inst_s, perc_s, idx_s = self._parse_esf_key(pair["esf_scan"])

            label = pair["label"]
            label = torch.tensor(label, dtype=torch.long)

            esf_r = self._get_esf(cls_r, inst_r, perc_r, idx_r)
            esf_s = self._get_esf(cls_s, inst_s, perc_s, idx_s)

            abs_diff = np.abs(esf_r - esf_s).reshape(10, 64)

            # Channels
            ch1 = self._normalize_local(abs_diff)        # shape 10x64
            ch2 = self._normalize_global(abs_diff)       # shape 10x64
            grad = np.abs(np.gradient(ch1, axis=1))      # shape 10x64
            ch_stack = np.stack([ch1, ch2, grad], axis=0).astype(np.float32)
            ch_tensor = torch.from_numpy(ch_stack)

            if self.use_metrics:
                # optional
                metrics = self._compute_metrics(esf_r, esf_s)
                self.preprocessed.append((ch_tensor, metrics, label))
            else:
                self.preprocessed.append((ch_tensor, label))

        print(f"✅ Preprocessing fertig: {len(self.preprocessed)} Samples")

    def __len__(self):
        return len(self.preprocessed)



    def _load_data(self):
        with open(self.esf_json_path, "r") as f:
            self.esf_data = json.load(f)

        with open(self.cv_info_json_path, "r") as f:
            cv = json.load(f)
        # cv[fold] enthält keys "train","val","test"
        split_info = cv[self.fold]

        # train/val/test Indizes
        self.train_pairs = split_info.get("train", [])
        self.val_pairs = split_info.get("val", [])
        self.test_pairs = split_info.get("test", [])

        # Alle paar-Daten (Dataset muss einheitlich arbeiten)
        self.all_pairs = self.train_pairs + self.val_pairs + self.test_pairs

        # speichere die indexbereiche
        self.train_indices = np.arange(0, len(self.train_pairs))
        self.val_indices = np.arange(len(self.train_pairs),
                                     len(self.train_pairs) + len(self.val_pairs))
        self.test_indices = np.arange(len(self.train_pairs) + len(self.val_pairs),
                                      len(self.all_pairs))

        print(f"✅ Fold {self.fold}: {len(self.train_pairs)} train, {len(self.val_pairs)} val, {len(self.test_pairs)} test")

    def __len__(self):
        return len(self.all_pairs)

    def _parse_esf_key(self, key_str: str):
        parts = key_str.split("_")
        if len(parts) < 4:
            raise ValueError(f"Ungültiges Key-Format: {key_str}")
        # alles außer den letzten drei Elementen ist der Klassenname
        cls = "_".join(parts[:-3])
        inst = parts[-3]
        perc = parts[-2]
        idx = int(parts[-1])
        return cls, inst, perc, idx

    def _get_esf(self, cls: str, inst: str, perc: str, idx: int):
        try:
            # durch die verschachtelten Strukturen navigieren
            esf_list = self.esf_data[cls][inst][perc]
            arr = np.array(esf_list[idx], dtype=np.float32)
            if arr.shape[0] != 640:
                raise ValueError(f"ESF shape invalid: {arr.shape}")
            return arr
        except KeyError as e:
            print(f"[WARN] Key not found in ESF dataset: {cls}/{inst}/{perc}/{idx} -> {e}")
            return np.zeros((640,), dtype=np.float32)
        except IndexError:
            print(f"[WARN] Index {idx} out of range for {cls}/{inst}/{perc}")
            return np.zeros((640,), dtype=np.float32)

    def _normalize_local(self, esf_flat: np.ndarray):
        reshaped = esf_flat.reshape(10, 64)
        max_vals = np.max(np.abs(reshaped), axis=1, keepdims=True)
        max_vals[max_vals < 1e-8] = 1.0
        return reshaped / max_vals  # shape [10,64]

    def _normalize_global(self, esf_flat: np.ndarray):
        max_val = np.max(np.abs(esf_flat))
        if max_val < 1e-8:
            return esf_flat.reshape(10, 64) * 0.0
        return (esf_flat / max_val).reshape(10, 64)

    def _compute_emd(self, desc1, desc2):
        emd_values = []
        for i in range(10):
            h1 = desc1[i]
            h2 = desc2[i]
            if np.all(h1 == 0) and np.all(h2 == 0):
                emd_values.append(0.0)
            else:
                try:
                    e = wasserstein_distance(h1, h2)
                    emd_values.append(float(np.clip(e, 0.0, 1.0)))
                except Exception:
                    emd_values.append(1.0)
        return emd_values

    def _compute_cosine_similarity(self, desc1, desc2):
        sims = []
        for i in range(10):
            h1 = desc1[i]
            h2 = desc2[i]
            if np.linalg.norm(h1) == 0 and np.linalg.norm(h2) == 0:
                sims.append(1.0)
            elif np.linalg.norm(h1) == 0 or np.linalg.norm(h2) == 0:
                sims.append(0.0)
            else:
                sim = 1 - cosine(h1, h2)
                sims.append((1 - sim) / 2.0)
        return sims

    def _getitem_backuip(self, idx):
        pair = self.all_pairs[idx]

        cls_ref, inst_ref, perc_ref, idx_ref = self._parse_esf_key(pair["esf_ref"])


        cls, inst, perc, idx = self._parse_esf_key(pair["esf_scan"])


        label = torch.tensor(pair["label"], dtype=torch.long)

        # Rohdaten holen
        esf_ref = self._get_esf(cls_ref, inst_ref, perc_ref, idx_ref)
        esf_scan = self._get_esf(cls, inst, perc, idx)
        # Differenz
        abs_diff = np.abs(esf_ref - esf_scan).reshape(10, 64)

        # Channel 1: lokal normiert
        ch1 = self._normalize_local(abs_diff.flatten())
        # Channel 2: global normiert
        ch2 = self._normalize_global(abs_diff.flatten())
        # Channel 3: gradientenbasiert (lokal normiert)
        grad = np.abs(np.gradient(ch1, axis=1))  # shape [10,64]

        channels = np.stack([ch1, ch2, grad], axis=0).astype(np.float32)  # [3,10,64]
        channels_tensor = torch.from_numpy(channels)

        if self.use_metrics:
            # Berechne zusätzlich globale Metriken (pro Kanal separat)
            ref_local = self._normalize_local(esf_ref)
            scan_local = self._normalize_local(esf_scan)
            ref_global = self._normalize_global(esf_ref)
            scan_global = self._normalize_global(esf_scan)
            ref_grad = np.abs(np.gradient(ref_local, axis=1))
            scan_grad = np.abs(np.gradient(scan_local, axis=1))

            emd_vals = self._compute_emd(ref_local, scan_local) + \
                       self._compute_emd(ref_global, scan_global) + \
                       self._compute_emd(ref_grad, scan_grad)
            cos_vals = self._compute_cosine_similarity(ref_local, scan_local) + \
                       self._compute_cosine_similarity(ref_global, scan_global) + \
                       self._compute_cosine_similarity(ref_grad, scan_grad)
            metrics = torch.tensor(emd_vals + cos_vals, dtype=torch.float32)  # 60 Werte
            return channels_tensor, metrics, label

        return channels_tensor, label

    def __getitem__(self, idx):
        return self.preprocessed[idx]

    def get_loaders(self, batch_size=32, batch_size_val=None, num_workers=4, persistent_workers=True):
        if num_workers > 0:
            self._persistent_workers = True
        else:
            self._persistent_workers = False

        val_bs = batch_size_val if batch_size_val is not None else batch_size

        train_loader = DataLoader(
            self,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(self.train_indices),
            pin_memory=True,
            num_workers=num_workers,
            persistent_workers=self._persistent_workers
        )

        val_loader = DataLoader(
            self,
            batch_size=val_bs,
            sampler=SubsetRandomSampler(self.val_indices),
            pin_memory=True,
            num_workers=num_workers,
            persistent_workers=self._persistent_workers
        )

        test_loader = DataLoader(
            self,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(self.test_indices),
            pin_memory=True,
            num_workers=num_workers,
            persistent_workers=self._persistent_workers
        )

        return train_loader, val_loader, test_loader


class ESFRefPairDataset_siamese(Dataset):
    def __init__(self, esf_data_path, cv_info_path, fold, extra_feats_path=None):
        # 🔥 ESF-Daten laden
        with open(esf_data_path, "rb") as f:
            self.esf_data = orjson.loads(f.read())

        # 🔥 CV-Infos laden
        with open(cv_info_path, "r") as f:
            cv_info = json.load(f)
        fold_info = cv_info[fold]

        # 🔥 Paare extrahieren
        self.train_pairs = fold_info.get("train", [])
        self.val_pairs   = fold_info.get("val", [])
        self.test_pairs  = fold_info.get("test", [])

        # 🔥 Indizes definieren
        self.train_indices = np.arange(0, len(self.train_pairs))
        self.val_indices   = np.arange(len(self.train_pairs),
                                       len(self.train_pairs)+len(self.val_pairs))
        self.test_indices  = np.arange(len(self.train_pairs)+len(self.val_pairs),
                                       len(self.train_pairs)+len(self.val_pairs)+len(self.test_pairs))

        # 🔥 Alle Paare zusammenführen
        self.all_pairs = self.train_pairs + self.val_pairs + self.test_pairs
        print(f"✅ Fold {fold}: {len(self.train_pairs)} train, {len(self.val_pairs)} val, {len(self.test_pairs)} test")

        # 🔥 Extra Features laden (sofern Pfad angegeben)
        self.extra_feats = None
        if extra_feats_path is not None:
            with open(extra_feats_path, "rb") as f:
                feats_data = orjson.loads(f.read())
            fold_feats = feats_data.get(fold, {})
            feats_train = fold_feats.get("train", [])
            feats_val   = fold_feats.get("val", [])
            feats_test  = fold_feats.get("test", [])
            # alle in gleiche Reihenfolge wie pairs
            self.extra_feats = feats_train + feats_val + feats_test
            if len(self.extra_feats) != len(self.all_pairs):
                print(f"[WARN] Anzahl extra_feats ({len(self.extra_feats)}) passt nicht zu Paare ({len(self.all_pairs)})")
            else:
                print(f"✅ Extra Features geladen: {len(self.extra_feats)}")

        # Cache
        self._esf_cache = {}

    def __len__(self):
        return len(self.all_pairs)

    def __getitem__(self, idx):
        pair = self.all_pairs[idx]

        # Keys zerlegen
        cls_r, inst_r, perc_r, idx_r = self._parse_esf_key(pair["esf_ref"])
        cls_s, inst_s, perc_s, idx_s = self._parse_esf_key(pair["esf_scan"])

        esf_ref = self._get_esf(cls_r, inst_r, perc_r, idx_r)
        esf_scan = self._get_esf(cls_s, inst_s, perc_s, idx_s)

        label = torch.tensor(pair["label"], dtype=torch.long)

        # falls extra_features vorhanden sind, per idx auslesen
        if self.extra_feats is not None and idx < len(self.extra_feats):
            extra_feat = np.array(self.extra_feats[idx], dtype=np.float32)
            extra_features = torch.from_numpy(extra_feat)
        else:
            extra_features = torch.zeros(0, dtype=torch.float32)

        return esf_ref, esf_scan, extra_features, label

    # ------------------------------------------------
    def _parse_esf_key(self, key_str: str):
        # z.B. "z_Campus_VAL_0_100_2"
        parts = key_str.split("_")
        if len(parts) < 4:
            raise ValueError(f"Ungültiger Key: {key_str}")
        cls = "_".join(parts[:-3])
        inst = parts[-3]
        perc = parts[-2]
        idx = int(parts[-1])
        return cls, inst, perc, idx

    def _get_esf(self, cls, inst, perc, idx):
        key = (cls, inst, perc, idx)
        if key in self._esf_cache:
            return self._esf_cache[key]
        try:
            vec = self.esf_data[cls][inst][perc][idx]
            arr = np.array(vec, dtype=np.float32)
            if arr.shape[0] != 640:
                raise ValueError(f"Shape ungültig: {arr.shape}")
            arr = self._normalize_esf_local_max(arr)
            tensor = torch.from_numpy(arr.reshape(1, 10, 64))
            self._esf_cache[key] = tensor
            return tensor
        except Exception as e:
            print(f"[WARN] Zugriff fehlgeschlagen: {cls}/{inst}/{perc}/{idx} -> {e}")
            return torch.zeros((1,10,64), dtype=torch.float32)

    def _normalize_esf_local_max(self, esf_flat):
        reshaped = esf_flat.reshape(10, 64)
        max_vals = np.max(np.abs(reshaped), axis=1, keepdims=True)
        max_vals[max_vals < 1e-8] = 1.0
        return (reshaped / max_vals).astype(np.float32).flatten()

    def get_loaders(self, batch_size=32, batch_size_val=None, num_workers=4):
        val_bs = batch_size_val if batch_size_val is not None else batch_size
        train_loader = DataLoader(
            self, batch_size=batch_size,
            sampler=SubsetRandomSampler(self.train_indices),
            pin_memory=True, num_workers=num_workers, persistent_workers=(num_workers>0)
        )
        val_loader = DataLoader(
            self, batch_size=val_bs,
            sampler=SubsetRandomSampler(self.val_indices),
            pin_memory=True, num_workers=num_workers, persistent_workers=(num_workers>0)
        )
        test_loader = DataLoader(
            self, batch_size=batch_size,
            sampler=SubsetRandomSampler(self.test_indices),
            pin_memory=True, num_workers=num_workers, persistent_workers=(num_workers>0)
        )
        return train_loader, val_loader, test_loader



class ESFRefPairDatasetChannels_Xfeats(Dataset):
    def __init__(self, esf_data_path, cv_info_path, fold,
                 use_metrics=False, extra_feats_path=None):
        self.use_metrics = use_metrics
        self.equal = False
        self._esf_cache = {}  # 🔥 Hier landen später die fertigen Samples

        # 🔹 ESF-Daten laden
        with open(esf_data_path, "rb") as f:
            self.esf_data = orjson.loads(f.read())

        # 🔹 Cross-Validation-Info laden
        with open(cv_info_path, "r") as f:
            cv_info = json.load(f)
        fold_info = cv_info[fold]

        # 🔹 Indexbereiche definieren
        self.train_pairs = fold_info.get("train", [])
        self.val_pairs = fold_info.get("val", [])
        self.test_pairs = fold_info.get("test", [])
        self.train_indices = np.arange(0, len(self.train_pairs))
        self.val_indices = np.arange(len(self.train_pairs),
                                     len(self.train_pairs) + len(self.val_pairs))
        self.test_indices = np.arange(len(self.train_pairs) + len(self.val_pairs),
                                      len(self.train_pairs) + len(self.val_pairs) + len(self.test_pairs))
        self.all_pairs = self.train_pairs + self.val_pairs + self.test_pairs

        print(f"✅ Fold {fold}: {len(self.train_pairs)} train, {len(self.val_pairs)} val, {len(self.test_pairs)} test")

        # 🔹 Optional: Extra Features laden
        self.extra_feats = None
        if extra_feats_path is not None:
            with open(extra_feats_path, "rb") as f:
                feats_data = orjson.loads(f.read())
            fold_feats = feats_data.get(fold, {})
            feats_train = fold_feats.get("train", [])
            feats_val = fold_feats.get("val", [])
            feats_test = fold_feats.get("test", [])
            self.extra_feats = feats_train + feats_val + feats_test
            if len(self.extra_feats) != len(self.all_pairs):
                print(f"[WARN] extra_feats({len(self.extra_feats)}) != pairs({len(self.all_pairs)})")
            else:
                print(f"✅ Extra Features geladen: {len(self.extra_feats)}")

        # ✅ Kein Preprocessing hier!
        #    → Erst in __getitem__ via _do_preprocessing, mit Cache.
        print("⏳ Preprocessing wird Lazy durchgeführt (on-demand).")


        if self.equal == True:

            # ---- Nur Training bearbeiten ----
            train_labels = [p["label"] for p in self.train_pairs]
            train_idx_0 = [i for i, lbl in enumerate(train_labels) if lbl == 0]
            train_idx_1 = [i for i, lbl in enumerate(train_labels) if lbl == 1]
            train_idx_2 = [i for i, lbl in enumerate(train_labels) if lbl == 2]

            # wie viele 2er sollen behalten werden?
            n1 = len(train_idx_1)
            if len(train_idx_2) > n1:
                train_idx_2 = random.sample(train_idx_2, n1)  # reduzieren auf Anzahl von Label 1

            # neue kombinierte Liste
            train_final_local = train_idx_0 + train_idx_1 + train_idx_2
            random.shuffle(train_final_local)

            # da Training immer am Anfang liegt, ist global = lokal
            self.train_indices = np.array(train_final_local)

            print(f"✅ Training balanciert (Label2 reduziert):")
            print(f"   Label0: {len(train_idx_0)}, Label1: {len(train_idx_1)}, Label2 (reduziert): {len(train_idx_2)}")
            print(f"   Gesamt train_indices: {len(self.train_indices)}")


            # # 🔹 Schritt 1: Labels für val sammeln
            # val_labels = [p["label"] for p in self.val_pairs]
            # val_idx_0 = [i for i, lbl in enumerate(val_labels) if lbl == 0]
            # val_idx_1 = [i for i, lbl in enumerate(val_labels) if lbl == 1]
            #
            # # 🔹 Schritt 2: gleiche Anzahl wählen
            # min_val = min(len(val_idx_0), len(val_idx_1))
            # val_idx_0_bal = random.sample(val_idx_0, min_val)
            # val_idx_1_bal = random.sample(val_idx_1, min_val)
            # val_balanced_indices_local = val_idx_0_bal + val_idx_1_bal
            # random.shuffle(val_balanced_indices_local)
            #
            # # 🔹 Schritt 3: Globale Indizes anpassen
            # val_offset = len(self.train_pairs)
            # self.val_indices = np.array([val_offset + i for i in val_balanced_indices_local])
            #
            # # 🔹 Schritt 4: Labels für test sammeln
            # test_labels = [p["label"] for p in self.test_pairs]
            # test_idx_0 = [i for i, lbl in enumerate(test_labels) if lbl == 0]
            # test_idx_1 = [i for i, lbl in enumerate(test_labels) if lbl == 1]
            #
            # # 🔹 Schritt 5: gleiche Anzahl wählen
            # min_test = min(len(test_idx_0), len(test_idx_1))
            # test_idx_0_bal = random.sample(test_idx_0, min_test)
            # test_idx_1_bal = random.sample(test_idx_1, min_test)
            # test_balanced_indices_local = test_idx_0_bal + test_idx_1_bal
            # random.shuffle(test_balanced_indices_local)
            #
            # # 🔹 Schritt 6: Globale Indizes anpassen
            # test_offset = len(self.train_pairs) + len(self.val_pairs)
            # self.test_indices = np.array([test_offset + i for i in test_balanced_indices_local])
            #
            # print(f"✅ Balanced val indices: {len(self.val_indices)} (0/1 gleichverteilt)")
            # print(f"✅ Balanced test indices: {len(self.test_indices)} (0/1 gleichverteilt)")

    def get_balanced_train_sampler(self):
        import random
        from torch.utils.data import SubsetRandomSampler

        train_labels = [p["label"] for p in self.train_pairs]
        idx_0 = [i for i, lbl in enumerate(train_labels) if lbl == 0]
        idx_1 = [i for i, lbl in enumerate(train_labels) if lbl == 1]
        idx_2 = [i for i, lbl in enumerate(train_labels) if lbl == 2]

        # 🔹 Anzahl 2er auf Anzahl 1er begrenzen
        n1 = len(idx_1)
        if len(idx_2) > n1:
            idx_2 = random.sample(idx_2, n1)

        # 🔹 Optional: auch 0er balancieren
        # idx_0 = random.sample(idx_0, n1)

        indices = idx_0 + idx_1 + idx_2
        random.shuffle(indices)
        return SubsetRandomSampler(indices)

    def _do_preprocessing(self, idx: int):
        # 🔹 1. Hole das Paar
        pair = self.all_pairs[idx]
        cls_r, inst_r, perc_r, idx_r = self._parse_esf_key(pair["esf_ref"])
        cls_s, inst_s, perc_s, idx_s = self._parse_esf_key(pair["esf_scan"])
        label = torch.tensor(pair["label"], dtype=torch.long)

        # 🔹 2. Lade die rohen ESF-Daten
        esf_r = self._get_esf(cls_r, inst_r, perc_r, idx_r)
        esf_s = self._get_esf(cls_s, inst_s, perc_s, idx_s)

        # 🔹 3. Feature-Berechnung
        abs_diff = np.abs(esf_r - esf_s).reshape(10, 64)
        if np.random.rand() < 0.8:
            abs_diff = self._augment_abs_diff(abs_diff)
        ch1 = self._normalize_local(abs_diff)
        ch2 = self._normalize_global(abs_diff)
        grad = np.abs(np.gradient(ch1, axis=1))
        ch_stack = np.stack([ch1, ch2, grad], axis=0).astype(np.float32)
        #ch_stack = ch2[np.newaxis, :, :].astype(np.float32)
        ch_tensor = torch.from_numpy(ch_stack)

        # 🔹 4. Extra Features verarbeiten
        if self.extra_feats is not None and idx < len(self.extra_feats):
            extra_vec = np.array(self.extra_feats[idx], dtype=np.float32)
            extra_tensor = torch.from_numpy(extra_vec)
        else:
            extra_tensor = torch.zeros(40, dtype=torch.float32)

        # 🔹 5. Option: spezielle Behandlung für bestimmte Labels
        if label == 2:
            return (ch_tensor, extra_tensor, torch.tensor(0, dtype=torch.long))

        return (ch_tensor, extra_tensor, label)

    def __len__(self):
        return len(self.preprocessed)

    def _augment_abs_diff(self, abs_diff):
        # Beispiel: leichte Skalierung + Rauschen + Shift
        #scale = np.random.uniform(0.95, 1.05)
        #abs_diff = abs_diff * scale
        noise = np.random.normal(0, 0.01, abs_diff.shape)
        abs_diff = abs_diff + noise
        #shift = np.random.randint(-2, 3)
        #abs_diff = np.roll(abs_diff, shift, axis=1)
        return abs_diff

    def __getitem__(self, idx):
        #if idx in self._esf_cache:
        #    return self._esf_cache[idx]
        # Preprocessing on-the-fly
        ch_tensor, extra_tensor, label = self._do_preprocessing(idx)
        return ch_tensor, extra_tensor, label
        #self._esf_cache[idx] = (ch_tensor, extra_tensor, label)
        #return self._esf_cache[idx]

    def _parse_esf_key(self, key_str: str):
        # z.B. "z_Campus_VAL_0_100_2"
        parts = key_str.split("_")
        if len(parts) < 4:
            raise ValueError(f"Ungültiger Key: {key_str}")
        cls = "_".join(parts[:-3])
        inst = parts[-3]
        perc = parts[-2]
        idx = int(parts[-1])
        return cls, inst, perc, idx

    def _normalize_local(self, esf_flat: np.ndarray):
        reshaped = esf_flat.reshape(10, 64)
        max_vals = np.max(np.abs(reshaped), axis=1, keepdims=True)
        max_vals[max_vals < 1e-8] = 1.0
        return reshaped / max_vals  # shape [10,64]

    def _normalize_global(self, esf_flat: np.ndarray):
        max_val = np.max(np.abs(esf_flat))
        if max_val < 1e-8:
            return esf_flat.reshape(10, 64) * 0.0
        return (esf_flat / max_val).reshape(10, 64)

    def _get_esf(self, cls, inst, perc, idx):
        key = (cls, inst, perc, idx)
        #if key in self._esf_cache:
        #    return self._esf_cache[key]
        try:
            vec = self.esf_data[cls][inst][perc][idx]
            arr = np.array(vec, dtype=np.float32)
            if arr.shape[0] != 640:
                raise ValueError(f"Shape ungültig: {arr.shape}")
            arr = arr.reshape(10,64)  # direkt richtig
            #self._esf_cache[key] = arr
            return arr
        except Exception as e:
            print(f"[WARN] Zugriff fehlgeschlagen: {cls}/{inst}/{perc}/{idx} -> {e}")
            return torch.zeros((1,10,64), dtype=torch.float32)


    def get_loaders(self, batch_size=32, batch_size_val=None, num_workers=4):
        val_bs = batch_size_val if batch_size_val is not None else batch_size
        train_loader = DataLoader(
            self, batch_size=batch_size,
            sampler=SubsetRandomSampler(self.train_indices),
            pin_memory=True, num_workers=num_workers, persistent_workers=(num_workers>0)
        )
        val_loader = DataLoader(
            self, batch_size=val_bs,
            sampler=SubsetRandomSampler(self.val_indices),
            pin_memory=True, num_workers=num_workers, persistent_workers=(num_workers>0)
        )
        test_loader = DataLoader(
            self, batch_size=batch_size,
            sampler=SubsetRandomSampler(self.test_indices),
            pin_memory=True, num_workers=num_workers, persistent_workers=(num_workers>0)
        )
        return train_loader, val_loader, test_loader



class ESFRefPairDatasetChannels_5(Dataset):
    def __init__(self, esf_data_path, cv_info_path, normal_hist_path, fold,
                 use_metrics=False, extra_feats_path=None):
        self.use_metrics = use_metrics
        self.equal = False
        self._esf_cache = {}  # 🔥 Hier landen später die fertigen Samples

        # 🔹 ESF-Daten laden
        with open(esf_data_path, "rb") as f:
            self.esf_data = orjson.loads(f.read())

            # 🔹 ESF-Daten laden
            with open(normal_hist_path, "rb") as f:
                self.normal_data = orjson.loads(f.read())

        # 🔹 Cross-Validation-Info laden
        with open(cv_info_path, "r") as f:
            cv_info = json.load(f)
        fold_info = cv_info[fold]

        # 🔹 Indexbereiche definieren
        self.train_pairs = fold_info.get("train", [])
        self.val_pairs = fold_info.get("val", [])
        self.test_pairs = fold_info.get("test", [])
        self.train_indices = np.arange(0, len(self.train_pairs))
        self.val_indices = np.arange(len(self.train_pairs),
                                     len(self.train_pairs) + len(self.val_pairs))
        self.test_indices = np.arange(len(self.train_pairs) + len(self.val_pairs),
                                      len(self.train_pairs) + len(self.val_pairs) + len(self.test_pairs))
        self.all_pairs = self.train_pairs + self.val_pairs + self.test_pairs

        print(f"✅ Fold {fold}: {len(self.train_pairs)} train, {len(self.val_pairs)} val, {len(self.test_pairs)} test")

        # 🔹 Optional: Extra Features laden
        self.extra_feats = None
        if extra_feats_path is not None:
            with open(extra_feats_path, "rb") as f:
                feats_data = orjson.loads(f.read())
            fold_feats = feats_data.get(fold, {})
            feats_train = fold_feats.get("train", [])
            feats_val = fold_feats.get("val", [])
            feats_test = fold_feats.get("test", [])
            self.extra_feats = feats_train + feats_val + feats_test
            if len(self.extra_feats) != len(self.all_pairs):
                print(f"[WARN] extra_feats({len(self.extra_feats)}) != pairs({len(self.all_pairs)})")
            else:
                print(f"✅ Extra Features geladen: {len(self.extra_feats)}")

        # ✅ Kein Preprocessing hier!
        #    → Erst in __getitem__ via _do_preprocessing, mit Cache.
        print("⏳ Preprocessing wird Lazy durchgeführt (on-demand).")


        if self.equal == True:

            # ---- Nur Training bearbeiten ----
            train_labels = [p["label"] for p in self.train_pairs]
            train_idx_0 = [i for i, lbl in enumerate(train_labels) if lbl == 0]
            train_idx_1 = [i for i, lbl in enumerate(train_labels) if lbl == 1]
            train_idx_2 = [i for i, lbl in enumerate(train_labels) if lbl == 2]

            # wie viele 2er sollen behalten werden?
            n1 = len(train_idx_1)
            if len(train_idx_2) > n1:
                train_idx_2 = random.sample(train_idx_2, n1)  # reduzieren auf Anzahl von Label 1

            # neue kombinierte Liste
            train_final_local = train_idx_0 + train_idx_1 + train_idx_2
            random.shuffle(train_final_local)

            # da Training immer am Anfang liegt, ist global = lokal
            self.train_indices = np.array(train_final_local)

            print(f"✅ Training balanciert (Label2 reduziert):")
            print(f"   Label0: {len(train_idx_0)}, Label1: {len(train_idx_1)}, Label2 (reduziert): {len(train_idx_2)}")
            print(f"   Gesamt train_indices: {len(self.train_indices)}")


            # # 🔹 Schritt 1: Labels für val sammeln
            # val_labels = [p["label"] for p in self.val_pairs]
            # val_idx_0 = [i for i, lbl in enumerate(val_labels) if lbl == 0]
            # val_idx_1 = [i for i, lbl in enumerate(val_labels) if lbl == 1]
            #
            # # 🔹 Schritt 2: gleiche Anzahl wählen
            # min_val = min(len(val_idx_0), len(val_idx_1))
            # val_idx_0_bal = random.sample(val_idx_0, min_val)
            # val_idx_1_bal = random.sample(val_idx_1, min_val)
            # val_balanced_indices_local = val_idx_0_bal + val_idx_1_bal
            # random.shuffle(val_balanced_indices_local)
            #
            # # 🔹 Schritt 3: Globale Indizes anpassen
            # val_offset = len(self.train_pairs)
            # self.val_indices = np.array([val_offset + i for i in val_balanced_indices_local])
            #
            # # 🔹 Schritt 4: Labels für test sammeln
            # test_labels = [p["label"] for p in self.test_pairs]
            # test_idx_0 = [i for i, lbl in enumerate(test_labels) if lbl == 0]
            # test_idx_1 = [i for i, lbl in enumerate(test_labels) if lbl == 1]
            #
            # # 🔹 Schritt 5: gleiche Anzahl wählen
            # min_test = min(len(test_idx_0), len(test_idx_1))
            # test_idx_0_bal = random.sample(test_idx_0, min_test)
            # test_idx_1_bal = random.sample(test_idx_1, min_test)
            # test_balanced_indices_local = test_idx_0_bal + test_idx_1_bal
            # random.shuffle(test_balanced_indices_local)
            #
            # # 🔹 Schritt 6: Globale Indizes anpassen
            # test_offset = len(self.train_pairs) + len(self.val_pairs)
            # self.test_indices = np.array([test_offset + i for i in test_balanced_indices_local])
            #
            # print(f"✅ Balanced val indices: {len(self.val_indices)} (0/1 gleichverteilt)")
            # print(f"✅ Balanced test indices: {len(self.test_indices)} (0/1 gleichverteilt)")

    def get_balanced_train_sampler(self):
        import random
        from torch.utils.data import SubsetRandomSampler

        train_labels = [p["label"] for p in self.train_pairs]
        idx_0 = [i for i, lbl in enumerate(train_labels) if lbl == 0]
        idx_1 = [i for i, lbl in enumerate(train_labels) if lbl == 1]
        idx_2 = [i for i, lbl in enumerate(train_labels) if lbl == 2]

        # 🔹 Anzahl 2er auf Anzahl 1er begrenzen
        n1 = len(idx_1)
        if len(idx_2) > n1:
            idx_2 = random.sample(idx_2, n1)

        # 🔹 Optional: auch 0er balancieren
        # idx_0 = random.sample(idx_0, n1)

        indices = idx_0 + idx_1 + idx_2
        random.shuffle(indices)
        return SubsetRandomSampler(indices)

    def _do_preprocessing(self, idx: int):
        # 1. Hole das Paar
        pair = self.all_pairs[idx]
        cls_r, inst_r, perc_r, idx_r = self._parse_esf_key(pair["esf_ref"])
        cls_s, inst_s, perc_s, idx_s = self._parse_esf_key(pair["esf_scan"])
        label = torch.tensor(pair["label"], dtype=torch.long)
        # 🔹 5. Option: spezielle Behandlung für bestimmte Labels


        # 2. Lade die rohen ESF-Daten und normal daten
        # esf_r = self._get_esf(cls_r, inst_r, perc_r, idx_r) #(10,64)
        # esf_s = self._get_esf(cls_s, inst_s, perc_s, idx_s)#(10,64)
        #
        # normal_r = self._get_esf(cls_r, inst_r, perc_r, idx_r) #(1,64)
        # normal_s = self._get_esf(cls_s, inst_s, perc_s, idx_s) #(10,64)

        esf_norm_r = self._get_esf_normal_combined(cls_r, inst_r, perc_r, idx_r)
        esf_norm_s = self._get_esf_normal_combined(cls_s, inst_s, perc_s, idx_s)

        # 3. Global normalisiertes abs_diff als Channel 1
        abs_diff = np.abs(esf_norm_r - esf_norm_s).reshape(11, 64)
        ch1 = self._normalize_global(abs_diff)  # (10,64)

        # 4. Extra Features aufsplitten in 4 Kanäle
        if self.extra_feats is not None and idx < len(self.extra_feats):
            extra_vec = np.array(self.extra_feats[idx], dtype=np.float32)
            if extra_vec.shape[0] != 40:
                raise ValueError(f"Extra Feats müssen 40 Werte haben, bekommen: {extra_vec.shape}")
        else:
            extra_vec = np.zeros(40, dtype=np.float32)

        # 5. Für jeden 10er-Block ein eigenes (10x64)-Feature bauen
        extra_channels = []
        for start in [0, 10, 20, 30]:
            block = extra_vec[start:start + 10].reshape(10, 1)  # (10,1)
            block = np.repeat(block, 64, axis=1).astype(np.float32)  # (10,64)
            extra_channels.append(block)

        # 6. Alle Kanäle stacken: 1x(10x64) + 4x(10x64) = 5x10x64
        ch_stack = np.stack([ch1] + extra_channels, axis=0).astype(np.float32)  # (5,10,64)
        ch_tensor = torch.from_numpy(ch_stack)


        if label == 2:
            return (ch_tensor, torch.tensor([]), torch.tensor(0, dtype=torch.long))

        return (ch_tensor, torch.tensor([]), label)  # extra_tensor entfällt hier

    def __len__(self):
        return len(self.preprocessed)

    def _augment_abs_diff(self, abs_diff):
        # Beispiel: leichte Skalierung + Rauschen + Shift
        #scale = np.random.uniform(0.95, 1.05)
        #abs_diff = abs_diff * scale
        noise = np.random.normal(0, 0.01, abs_diff.shape)
        abs_diff = abs_diff + noise
        #shift = np.random.randint(-2, 3)
        #abs_diff = np.roll(abs_diff, shift, axis=1)
        return abs_diff

    def __getitem__(self, idx):
        #if idx in self._esf_cache:
        #    return self._esf_cache[idx]
        # Preprocessing on-the-fly
        ch_tensor, extra_tensor, label = self._do_preprocessing(idx)
        #self._esf_cache[idx] = (ch_tensor, extra_tensor, label)
        return ch_tensor, extra_tensor, label

    def _parse_esf_key(self, key_str: str):
        # z.B. "z_Campus_VAL_0_100_2"
        parts = key_str.split("_")
        if len(parts) < 4:
            raise ValueError(f"Ungültiger Key: {key_str}")
        cls = "_".join(parts[:-3])
        inst = parts[-3]
        perc = parts[-2]
        idx = int(parts[-1])
        return cls, inst, perc, idx

    def _normalize_local(self, esf_flat: np.ndarray):
        reshaped = esf_flat.reshape(10, 64)
        max_vals = np.max(np.abs(reshaped), axis=1, keepdims=True)
        max_vals[max_vals < 1e-8] = 1.0
        return reshaped / max_vals  # shape [10,64]

    def _normalize_global(self, esf_flat: np.ndarray):
        max_val = np.max(np.abs(esf_flat))
        if max_val < 1e-8:
            return esf_flat.reshape(10, 64) * 0.0
        return (esf_flat / max_val).reshape(10, 64)

    def _get_esf(self, cls, inst, perc, idx):
        key = (cls, inst, perc, idx)
        #if key in self._esf_cache:
        #    return self._esf_cache[key]
        try:
            vec = self.esf_data[cls][inst][perc][idx]
            arr = np.array(vec, dtype=np.float32)
            if arr.shape[0] != 640:
                raise ValueError(f"Shape ungültig: {arr.shape}")
            arr = arr.reshape(10,64)  # direkt richtig
            #self._esf_cache[key] = arr
            return arr
        except Exception as e:
            print(f"[WARN] Zugriff fehlgeschlagen: {cls}/{inst}/{perc}/{idx} -> {e}")
            return torch.zeros((1,10,64), dtype=torch.float32)

    def _get_normal(self, cls, inst, perc, idx):
        key = (cls, inst, perc, idx)
        #if key in self._esf_cache:
        #    return self._esf_cache[key]
        try:
            vec = self.normal_data[cls][inst][perc][idx]
            arr = np.array(vec, dtype=np.float32)
            if arr.shape[0] != 64:
                raise ValueError(f"Shape ungültig: {arr.shape}")
            arr = arr.reshape(1,64)  # direkt richtig
            #self._esf_cache[key] = arr
            return arr
        except Exception as e:
            print(f"[WARN] Zugriff fehlgeschlagen: {cls}/{inst}/{perc}/{idx} -> {e}")
            return torch.zeros((1,1,64), dtype=torch.float32)

    def _get_esf_normal_combined(self, cls, inst, perc, idx):
        try:
            # ESF laden
            esf_vec = self.esf_data[cls][inst][perc][idx]
            esf_arr = np.array(esf_vec, dtype=np.float32)
            if esf_arr.shape[0] != 640:
                raise ValueError(f"Ungültige ESF-Shape: {esf_arr.shape}")
            esf_arr = esf_arr.reshape(10, 64)

            # Normalen laden
            normal_vec = self.normal_data[cls][inst][perc][idx]
            normal_arr = np.array(normal_vec, dtype=np.float32)
            if normal_arr.shape[0] != 64:
                raise ValueError(f"Ungültige Normalen-Shape: {normal_arr.shape}")
            normal_arr = normal_arr.reshape(1, 64)

            # Kombinieren → (11, 64)
            combined = np.concatenate([esf_arr, normal_arr], axis=0)
            return combined  # shape (11, 64)

        except Exception as e:
            print(f"[SKIP] Fehler bei {cls}/{inst}/{perc}/{idx} -> {e}")
            return None


    def get_loaders(self, batch_size=32, batch_size_val=None, num_workers=4):
        val_bs = batch_size_val if batch_size_val is not None else batch_size
        train_loader = DataLoader(
            self, batch_size=batch_size,
            sampler=SubsetRandomSampler(self.train_indices),
            pin_memory=True, num_workers=num_workers, persistent_workers=(num_workers>0)
        )
        val_loader = DataLoader(
            self, batch_size=val_bs,
            sampler=SubsetRandomSampler(self.val_indices),
            pin_memory=True, num_workers=num_workers, persistent_workers=(num_workers>0)
        )
        test_loader = DataLoader(
            self, batch_size=batch_size,
            sampler=SubsetRandomSampler(self.test_indices),
            pin_memory=True, num_workers=num_workers, persistent_workers=(num_workers>0)
        )
        return train_loader, val_loader, test_loader



class ESFRefPairDatasetChannels_4(Dataset):
    def __init__(self, esf_data_path, cv_info_path, normal_hist_path, fold,
                 use_metrics=False, extra_feats_path=None):
        self.use_metrics = use_metrics
        self.equal = False
        self._esf_cache = {}  # 🔥 Hier landen später die fertigen Samples

        # 🔹 ESF-Daten laden
        with open(esf_data_path, "rb") as f:
            self.esf_data = orjson.loads(f.read())

            # 🔹 ESF-Daten laden
            with open(normal_hist_path, "rb") as f:
                self.normal_data = orjson.loads(f.read())

        # 🔹 Cross-Validation-Info laden
        with open(cv_info_path, "r") as f:
            cv_info = json.load(f)
        fold_info = cv_info[fold]

        # 🔹 Paare definieren
        raw_train_pairs = fold_info.get("train", [])
        raw_val_pairs = fold_info.get("val", [])
        raw_test_pairs = fold_info.get("test", [])

        # 🔸 Extra-Features laden und aufteilen
        feats_train = feats_val = feats_test = None
        if extra_feats_path is not None:
            with open(extra_feats_path, "rb") as f:
                feats_data = orjson.loads(f.read())
            fold_feats = feats_data.get(fold, {})
            feats_train = fold_feats.get("train", [])
            feats_val = fold_feats.get("val", [])
            feats_test = fold_feats.get("test", [])

        # 🔹 Filterung der gültigen Paare
        self.train_pairs, feats_train = self._filter_valid_pairs(raw_train_pairs, feats_train)
        self.val_pairs, feats_val = self._filter_valid_pairs(raw_val_pairs, feats_val)
        self.test_pairs, feats_test = self._filter_valid_pairs(raw_test_pairs, feats_test)

        # 🔹 Zusammenführen
        self.all_pairs = self.train_pairs + self.val_pairs + self.test_pairs
        self.extra_feats = feats_train + feats_val + feats_test if feats_train else None

        # 🔹 Indexbereiche berechnen
        self.train_indices = np.arange(0, len(self.train_pairs))
        self.val_indices = np.arange(len(self.train_pairs), len(self.train_pairs) + len(self.val_pairs))
        self.test_indices = np.arange(len(self.train_pairs) + len(self.val_pairs), len(self.all_pairs))

        print(f"✅ Train: {len(self.train_pairs)}, Val: {len(self.val_pairs)}, Test: {len(self.test_pairs)} (nur gültige)")

        print(f"✅ Fold {fold}: {len(self.train_pairs)} train, {len(self.val_pairs)} val, {len(self.test_pairs)} test")

        # ✅ Kein Preprocessing hier!
        #    → Erst in __getitem__ via _do_preprocessing, mit Cache.
        print("⏳ Preprocessing wird Lazy durchgeführt (on-demand).")


        if self.equal == True:

            # ---- Nur Training bearbeiten ----
            train_labels = [p["label"] for p in self.train_pairs]
            train_idx_0 = [i for i, lbl in enumerate(train_labels) if lbl == 0]
            train_idx_1 = [i for i, lbl in enumerate(train_labels) if lbl == 1]
            train_idx_2 = [i for i, lbl in enumerate(train_labels) if lbl == 2]

            # wie viele 2er sollen behalten werden?
            n1 = len(train_idx_1)
            if len(train_idx_2) > n1:
                train_idx_2 = random.sample(train_idx_2, n1)  # reduzieren auf Anzahl von Label 1

            # neue kombinierte Liste
            train_final_local = train_idx_0 + train_idx_1 + train_idx_2
            random.shuffle(train_final_local)

            # da Training immer am Anfang liegt, ist global = lokal
            self.train_indices = np.array(train_final_local)

            print(f"✅ Training balanciert (Label2 reduziert):")
            print(f"   Label0: {len(train_idx_0)}, Label1: {len(train_idx_1)}, Label2 (reduziert): {len(train_idx_2)}")
            print(f"   Gesamt train_indices: {len(self.train_indices)}")


            # # 🔹 Schritt 1: Labels für val sammeln
            # val_labels = [p["label"] for p in self.val_pairs]
            # val_idx_0 = [i for i, lbl in enumerate(val_labels) if lbl == 0]
            # val_idx_1 = [i for i, lbl in enumerate(val_labels) if lbl == 1]
            #
            # # 🔹 Schritt 2: gleiche Anzahl wählen
            # min_val = min(len(val_idx_0), len(val_idx_1))
            # val_idx_0_bal = random.sample(val_idx_0, min_val)
            # val_idx_1_bal = random.sample(val_idx_1, min_val)
            # val_balanced_indices_local = val_idx_0_bal + val_idx_1_bal
            # random.shuffle(val_balanced_indices_local)
            #
            # # 🔹 Schritt 3: Globale Indizes anpassen
            # val_offset = len(self.train_pairs)
            # self.val_indices = np.array([val_offset + i for i in val_balanced_indices_local])
            #
            # # 🔹 Schritt 4: Labels für test sammeln
            # test_labels = [p["label"] for p in self.test_pairs]
            # test_idx_0 = [i for i, lbl in enumerate(test_labels) if lbl == 0]
            # test_idx_1 = [i for i, lbl in enumerate(test_labels) if lbl == 1]
            #
            # # 🔹 Schritt 5: gleiche Anzahl wählen
            # min_test = min(len(test_idx_0), len(test_idx_1))
            # test_idx_0_bal = random.sample(test_idx_0, min_test)
            # test_idx_1_bal = random.sample(test_idx_1, min_test)
            # test_balanced_indices_local = test_idx_0_bal + test_idx_1_bal
            # random.shuffle(test_balanced_indices_local)
            #
            # # 🔹 Schritt 6: Globale Indizes anpassen
            # test_offset = len(self.train_pairs) + len(self.val_pairs)
            # self.test_indices = np.array([test_offset + i for i in test_balanced_indices_local])
            #
            # print(f"✅ Balanced val indices: {len(self.val_indices)} (0/1 gleichverteilt)")
            # print(f"✅ Balanced test indices: {len(self.test_indices)} (0/1 gleichverteilt)")

    def _filter_valid_pairs(self, pairs, extra_feats=None):
        valid_pairs = []
        valid_feats = []

        for i, pair in enumerate(pairs):
            cls_r, inst_r, perc_r, idx_r = self._parse_esf_key(pair["esf_ref"])
            cls_s, inst_s, perc_s, idx_s = self._parse_esf_key(pair["esf_scan"])

            arr_r = self._get_esf_normal_combined(cls_r, inst_r, perc_r, idx_r)
            arr_s = self._get_esf_normal_combined(cls_s, inst_s, perc_s, idx_s)

            if (arr_r is not None and arr_s is not None
                    and arr_r.shape == (11, 64) and arr_s.shape == (11, 64)):
                valid_pairs.append(pair)
                if extra_feats is not None:
                    valid_feats.append(extra_feats[i])

        return valid_pairs, valid_feats if extra_feats is not None else None

    def get_balanced_train_sampler(self):
        import random
        from torch.utils.data import SubsetRandomSampler

        train_labels = [p["label"] for p in self.train_pairs]
        idx_0 = [i for i, lbl in enumerate(train_labels) if lbl == 0]
        idx_1 = [i for i, lbl in enumerate(train_labels) if lbl == 1]
        idx_2 = [i for i, lbl in enumerate(train_labels) if lbl == 2]

        # 🔹 Anzahl 2er auf Anzahl 1er begrenzen
        n1 = len(idx_1)
        if len(idx_2) > n1:
            idx_2 = random.sample(idx_2, n1)

        # 🔹 Optional: auch 0er balancieren
        # idx_0 = random.sample(idx_0, n1)

        indices = idx_0 + idx_1 + idx_2
        random.shuffle(indices)
        return SubsetRandomSampler(indices)

    def _do_preprocessing(self, idx: int):
        # 1. Hole das Paar
        extra_feats =  self.extra_feats[idx]
        pair = self.all_pairs[idx]
        cls_r, inst_r, perc_r, idx_r = self._parse_esf_key(pair["esf_ref"])
        cls_s, inst_s, perc_s, idx_s = self._parse_esf_key(pair["esf_scan"])
        label = torch.tensor(pair["label"], dtype=torch.long)
        # 🔹 5. Option: spezielle Behandlung für bestimmte Labels


        # 2. Lade die rohen ESF-Daten und normal daten
        # esf_r = self._get_esf(cls_r, inst_r, perc_r, idx_r) #(10,64)
        # esf_s = self._get_esf(cls_s, inst_s, perc_s, idx_s)#(10,64)
        #
        # normal_r = self._get_esf(cls_r, inst_r, perc_r, idx_r) #(1,64)
        # normal_s = self._get_esf(cls_s, inst_s, perc_s, idx_s) #(10,64)

        esf_norm_r = self._get_esf_normal_combined(cls_r, inst_r, perc_r, idx_r)
        esf_norm_s = self._get_esf_normal_combined(cls_s, inst_s, perc_s, idx_s)

        # 🔍 Fehlerbehandlung: Wenn Daten fehlen oder Shape falsch
        if (
                esf_norm_r is None or esf_norm_s is None
                or esf_norm_r.shape != (11, 64)
                or esf_norm_s.shape != (11, 64)
        ):
            return None  # Dieser Index wird vom Aufrufer übersprungen

        # 3. Global normalisiertes abs_diff als Channel 1
        # 🔹 1. Volle Differenz (ESF + Normal-Hist) → shape (11, 64)
        abs_diff = np.abs(esf_norm_r - esf_norm_s)

        # 🔹 2. Normal-Hist (letzte Zeile) → augmentieren mit Rotation
        normal_hist_2d = abs_diff[-1].reshape(8, 8)
        if np.random.rand() < 0.5:
            normal_hist_flat = self._augment_normal_hist_rotation(normal_hist_2d)
            normal_hist = np.array(normal_hist_flat).reshape(1, 64)
        else:
            normal_hist = abs_diff[-1].reshape(1, 64)

        # 🔹 3. Restliche 10 Zeilen (ESF) übernehmen
        esf_part = abs_diff[:-1]  # shape (10, 64)

        # 🔹 4. Neues abs_diff zusammensetzen → (11, 64)
        abs_diff = np.concatenate([esf_part, normal_hist], axis=0)
        if np.random.rand() < 0.5:
            abs_diff = self._augment_abs_diff(abs_diff)

        ch1 = self._normalize_local(abs_diff).astype(np.float32)
        ch2 = self._normalize_global(abs_diff).astype(np.float32)
        grad = np.abs(np.gradient(ch1, axis=1)).astype(np.float32)
        grad2 = np.abs(np.gradient(ch2, axis=1)).astype(np.float32)
        ch_stack = np.stack([ch1, grad2], axis=0).astype(np.float32)
        #ch_stack = ch2[np.newaxis, :, :].astype(np.float32)
        ch_tensor = torch.from_numpy(ch_stack)  # (4, 11, 64)

        # 4. Extra Features aufsplitten in 4 Kanäle

        if label == 2:
            return (ch_tensor, torch.tensor([]), torch.tensor(0, dtype=torch.long))

        return (ch_tensor, torch.tensor([]), label)  # extra_tensor entfällt hier

    def _augment_normal_hist_rotation(self, normal_hist: list, max_shift=7) -> list:
        hist = np.array(normal_hist).reshape(8, 8)
        shift = np.random.randint(1, max_shift + 1)
        hist_rotated = np.roll(hist, shift=shift, axis=0)
        return hist_rotated.flatten(order='C').tolist()

    def __len__(self):
        return len(self.preprocessed)

    def _augment_abs_diff(self, abs_diff):
        # Beispiel: leichte Skalierung + Rauschen + Shift
        #scale = np.random.uniform(0.95, 1.05)
        #abs_diff = abs_diff * scale
        noise = np.random.normal(0, 0.01, abs_diff.shape)
        abs_diff = abs_diff + noise
        #shift = np.random.randint(-2, 3)
        #abs_diff = np.roll(abs_diff, shift, axis=1)
        return abs_diff

    def __getitem__(self, idx):
        #if idx in self._esf_cache:
        #    return self._esf_cache[idx]
        # Preprocessing on-the-fly
        ch_tensor, extra_tensor, label = self._do_preprocessing(idx)
        #self._esf_cache[idx] = (ch_tensor, extra_tensor, label)
        return ch_tensor, extra_tensor, label

    def _parse_esf_key(self, key_str: str):
        # z.B. "z_Campus_VAL_0_100_2"
        parts = key_str.split("_")
        if len(parts) < 4:
            raise ValueError(f"Ungültiger Key: {key_str}")
        cls = "_".join(parts[:-3])
        inst = parts[-3]
        perc = parts[-2]
        idx = int(parts[-1])
        return cls, inst, perc, idx

    def _normalize_local(self, esf_flat: np.ndarray):
        reshaped = esf_flat.reshape(11, 64)
        max_vals = np.max(np.abs(reshaped), axis=1, keepdims=True)
        max_vals[max_vals < 1e-8] = 1.0
        return reshaped / max_vals  # shape [10,64]

    def _normalize_global(self, esf_flat: np.ndarray):
        max_val = np.max(np.abs(esf_flat))
        if max_val < 1e-8:
            return esf_flat.reshape(11, 64) * 0.0
        return (esf_flat / max_val).reshape(11, 64)

    def _get_esf(self, cls, inst, perc, idx):
        key = (cls, inst, perc, idx)
        #if key in self._esf_cache:
        #    return self._esf_cache[key]
        try:
            vec = self.esf_data[cls][inst][perc][idx]
            arr = np.array(vec, dtype=np.float32)
            if arr.shape[0] != 640:
                raise ValueError(f"Shape ungültig: {arr.shape}")
            arr = arr.reshape(10,64)  # direkt richtig
            #self._esf_cache[key] = arr
            return arr
        except Exception as e:
            print(f"[WARN] Zugriff fehlgeschlagen: {cls}/{inst}/{perc}/{idx} -> {e}")
            return torch.zeros((1,10,64), dtype=torch.float32)

    def _get_normal(self, cls, inst, perc, idx):
        key = (cls, inst, perc, idx)
        #if key in self._esf_cache:
        #    return self._esf_cache[key]
        try:
            vec = self.normal_data[cls][inst][perc][idx]
            arr = np.array(vec, dtype=np.float32)
            if arr.shape[0] != 64:
                raise ValueError(f"Shape ungültig: {arr.shape}")
            arr = arr.reshape(1,64)  # direkt richtig
            #self._esf_cache[key] = arr
            return arr
        except Exception as e:
            print(f"[WARN] Zugriff fehlgeschlagen: {cls}/{inst}/{perc}/{idx} -> {e}")
            return torch.zeros((1,1,64), dtype=torch.float32)

    def _get_esf_normal_combined(self, cls, inst, perc, idx):
        try:
            # ESF laden
            esf_vec = self.esf_data[cls][inst][perc][idx]
            esf_arr = np.array(esf_vec, dtype=np.float32)
            if esf_arr.shape[0] != 640:
                raise ValueError(f"Ungültige ESF-Shape: {esf_arr.shape}")
            esf_arr = esf_arr.reshape(10, 64)

            # Normalen laden
            normal_vec = self.normal_data[cls][inst][perc][idx]
            normal_arr = np.array(normal_vec, dtype=np.float32)
            if normal_arr.shape[0] != 64:
                raise ValueError(f"Ungültige Normalen-Shape: {normal_arr.shape}")
            normal_arr = normal_arr.reshape(1, 64)

            # Kombinieren → (11, 64)
            combined = np.concatenate([esf_arr, normal_arr], axis=0)
            return combined  # shape (11, 64)

        except Exception as e:
            print(f"[SKIP] Fehler bei {cls}/{inst}/{perc}/{idx} -> {e}")
            return None


    def get_loaders(self, batch_size=32, batch_size_val=None, num_workers=4):
        val_bs = batch_size_val if batch_size_val is not None else batch_size
        train_loader = DataLoader(
            self, batch_size=batch_size,
            sampler=SubsetRandomSampler(self.train_indices),
            pin_memory=True, num_workers=num_workers, persistent_workers=(num_workers>0)
        )
        val_loader = DataLoader(
            self, batch_size=val_bs,
            sampler=SubsetRandomSampler(self.val_indices),
            pin_memory=True, num_workers=num_workers, persistent_workers=(num_workers>0)
        )
        test_loader = DataLoader(
            self, batch_size=batch_size,
            sampler=SubsetRandomSampler(self.test_indices),
            pin_memory=True, num_workers=num_workers, persistent_workers=(num_workers>0)
        )
        return train_loader, val_loader, test_loader

import json, random, orjson, numpy as np, torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from scipy.spatial.distance import cosine as _cos
from scipy.stats import wasserstein_distance
from functools import lru_cache


class ESFRefPairDatasetChannels_4_Xfeat_backup(Dataset):
    # -------------------------------------------------- #
    #  Initialisierung
    # -------------------------------------------------- #
    def __init__(self, esf_path, normal_path, cv_path,
                 fold: str, use_metrics=False, extra_feats_path=None):
        self.use_metrics = use_metrics            # online-Features berechnen?
        self._load_raw_data(esf_path, normal_path)
        self._init_cv(cv_path, fold)
        self._load_precomputed_feats(extra_feats_path, fold)
        self._build_indices()
        print(f"✓ Dataset ready | Train {len(self.train_pairs)} | "
              f"Val {len(self.val_pairs)} | Test {len(self.test_pairs)}")

    # -------------------------------------------------- #
    #  Daten laden
    # -------------------------------------------------- #
    def _load_raw_data(self, esf_path, normal_path):
        with open(esf_path,  "rb") as f: self.esf = orjson.loads(f.read())
        with open(normal_path, "rb") as f: self.norm = orjson.loads(f.read())

    def _init_cv(self, cv_path, fold):
        with open(cv_path, "r") as f: cv = json.load(f)[fold]
        self.train_raw, self.val_raw, self.test_raw = (
            cv["train"], cv["val"], cv["test"])

    def _load_precomputed_feats(self, path, fold):
        if path is None:
            self.pre_feats = None
            return
        with open(path, "rb") as f: data = orjson.loads(f.read())
        fld = data.get(fold, {})
        self.pre_feats = (fld.get("train", []) +
                          fld.get("val",   []) +
                          fld.get("test",  []))

    # -------------------------------------------------- #
    #  Hilfs-Funktionen
    # -------------------------------------------------- #
    @staticmethod
    def _parse_key(key):
        p = key.split("_");  return "_".join(p[:-3]), p[-3], p[-2], int(p[-1])

    def _get_vec(self, store, cls, inst, perc, idx, exp_len):
        try:
            vec = store[cls][str(inst)][str(perc)][idx]
            arr = np.asarray(vec, np.float32)
            return arr if arr.size == exp_len else None
        except Exception:
            return None

    def _get_combined(self, cls, inst, perc, idx):
        e = self._get_vec(self.esf,  cls, inst, perc, idx, 640)
        n = self._get_vec(self.norm, cls, inst, perc, idx,  64)
        if e is None or n is None: return None
        return np.concatenate([e.reshape(10, 64), n.reshape(1, 64)], 0)  # (11,64)

    # 11×4-Feature-Block
    @staticmethod
    def _row_metrics(v1, v2):
        cos_local = 1 - _cos(v1, v2)
        v1n = v1 / (np.linalg.norm(v1) + 1e-8)
        v2n = v2 / (np.linalg.norm(v2) + 1e-8)
        cos_global = 1 - _cos(v1n, v2n)
        emd_local  = wasserstein_distance(v1, v2) / 64.0
        emd_global = wasserstein_distance(v1n, v2n) / 64.0
        return cos_local, cos_global, emd_local, emd_global

    # Augmentierungen ---------------------------------------------------- #
    @staticmethod
    def _aug_norm_rot(mat8x8):
        s = np.random.randint(1, 8)
        return np.roll(mat8x8, s, axis=0).reshape(1, 64)

    @staticmethod
    def _aug_noise(arr):
        return arr + np.random.normal(0, 0.01, arr.shape)

    # -------------------------------------------------- #
    #  Paar-Filterung & Indizes
    # -------------------------------------------------- #
    def _filter_pairs(self, raw_pairs):
        ok = []
        for p in raw_pairs:
            cls_r, inst_r, perc_r, idx_r = self._parse_key(p["esf_ref"])
            cls_s, inst_s, perc_s, idx_s = self._parse_key(p["esf_scan"])
            a = self._get_combined(cls_r, inst_r, perc_r, idx_r)
            b = self._get_combined(cls_s, inst_s, perc_s, idx_s)
            if a is not None and b is not None:
                ok.append(p)
        return ok

    def _build_indices(self):
        self.train_pairs = self._filter_pairs(self.train_raw)
        self.val_pairs   = self._filter_pairs(self.val_raw)
        self.test_pairs  = self._filter_pairs(self.test_raw)
        self.all_pairs   = self.train_pairs + self.val_pairs + self.test_pairs
        self.train_idx = np.arange(0, len(self.train_pairs))
        self.val_idx   = np.arange(len(self.train_pairs),
                                   len(self.train_pairs)+len(self.val_pairs))
        self.test_idx  = np.arange(len(self.all_pairs)-len(self.test_pairs),
                                   len(self.all_pairs))

    # -------------------------------------------------- #
    #  Haupt-Preprocessing-Pipeline
    # -------------------------------------------------- #
    def _do_preprocessing(self, i):
        pair = self.all_pairs[i]
        cls_r, inst_r, perc_r, idx_r = self._parse_key(pair["esf_ref"])
        cls_s, inst_s, perc_s, idx_s = self._parse_key(pair["esf_scan"])
        #label = torch.tensor(pair["label"], dtype=torch.long)
        y= pair["label"]
        if y == 0:
            return None  # raus damit
        y = 0 if y == 2 else 1
        label = torch.tensor(y, dtype=torch.long)

        A = self._get_combined(cls_r, inst_r, perc_r, idx_r)  # (11,64)
        B = self._get_combined(cls_s, inst_s, perc_s, idx_s)
        if (
                A is None or B is None or
                A.shape != (11, 64) or B.shape != (11, 64)
        ):
            return None

        # -------- Augmentierung --------
        diff = np.abs(A - B)                        # (11,64)
        if np.random.rand() < .5:                   # rotate nur Normal-Hist
            diff[-1:] = self._aug_norm_rot(diff[-1].reshape(8, 8))
        if np.random.rand() < .5:                   # Noise auf allem
            diff = self._aug_noise(diff)

        # -------- Kanäle --------
        ch1 = diff / (np.max(np.abs(diff), axis=1, keepdims=True)+1e-8)
        ch2 = diff / (np.max(np.abs(diff))+1e-8)
        grad = np.abs(np.gradient(ch1, axis=1))
        grad2 = np.abs(np.gradient(ch2, axis=1))
        stack = np.stack([ch1, ch2, grad, grad2]).astype(np.float32)  # (2,11,64)
        ch_tensor = torch.from_numpy(stack)

        # -------- Extra-Features --------
        if self.use_metrics:               # online berechnen
            feats = []
            for r in range(11):
                feats.extend(self._row_metrics(A[r], B[r]))
            # leichtes Rauschen
            if np.random.rand() < .3:
                feats += np.random.normal(0, .01, len(feats))
            extra_tensor = torch.tensor(feats, dtype=torch.float32)
        else:                              # vorgefertigte nutzen
            if self.pre_feats is None:
                extra_tensor = torch.tensor([])
            else:
                extra_tensor = torch.tensor(self.pre_feats[i],
                                              dtype=torch.float32)
        if label == 2:
            return (ch_tensor, extra_tensor, torch.tensor(0, dtype=torch.long))

        return ch_tensor, extra_tensor, label

    # -------------------------------------------------- #
    #  PyTorch-Dataset-API
    # -------------------------------------------------- #
    def __len__(self): return len(self.all_pairs)

    def __getitem__(self, idx):
        # Robust: springt weiter, bis ein gültiges Sample kommt
        while True:
            res = self._do_preprocessing(idx)
            if res is not None: return res
            idx = (idx + 1) % len(self.all_pairs)

    # -------------------------------------------------- #
    #  Loader-Helper
    # -------------------------------------------------- #
    def get_loaders(self, batch_size=32, batch_size_val=None, num_workers=0):
        """
        Liefert (train_loader, val_loader, test_loader).
        Passt zu: get_loaders(batch_size=..., batch_size_val=..., num_workers=...)
        """
        # Indizes robust holen (je nach Attributnamen)
        train_idx = getattr(self, "train_indices", getattr(self, "train_idx"))
        val_idx = getattr(self, "val_indices", getattr(self, "val_idx"))
        test_idx = getattr(self, "test_indices", getattr(self, "test_idx"))

        val_bs = batch_size_val if batch_size_val is not None else batch_size
        pw = bool(num_workers > 0)  # nur True erlauben, wenn >0 (Windows!)

        train_loader = DataLoader(
            self,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(train_idx),
            pin_memory=True,
            num_workers=num_workers,
            persistent_workers=pw,
            drop_last=False,
        )
        val_loader = DataLoader(
            self,
            batch_size=val_bs,
            sampler=SubsetRandomSampler(val_idx),
            pin_memory=True,
            num_workers=num_workers,
            persistent_workers=pw,
            drop_last=False,
        )
        test_loader = DataLoader(
            self,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(test_idx),
            pin_memory=True,
            num_workers=num_workers,
            persistent_workers=pw,
            drop_last=False,
        )
        return train_loader, val_loader, test_loader



class ESFRefPairDatasetChannels_4_Xfeat_toobig(Dataset):
    """
        Lädt ESF/NORM-Rohdaten, CV6-Paare und optionale, ausgerichtete Extra-Features.
        - Extra-Features: eine JSON/JSON.GZ Datei im Format
          {"foldX":{"train":[...], "val":[...], "test":[...]}, ...}
          *Gleiche Reihenfolge und Länge wie cv6_info.json.*
        - Die Klasse filtert Paare wie bisher.
        - Die Extra-Features werden per Maske auf die gekeepte Menge komprimiert.
        - Ergebnis: self.pre_feats[i] passt exakt zu self.all_pairs[i].
        """

    # -------------------------------------------------- #
    #  Initialisierung
    # -------------------------------------------------- #
    def __init__(self, esf_path, normal_path, cv_path,
                 fold: str, use_metrics=False, extra_feats_path=None):
        self.use_metrics = use_metrics  # online-Features berechnen?
        self._load_raw_data(esf_path, normal_path)
        self._init_cv(cv_path, fold)
        self._load_precomputed_feats(extra_feats_path, fold)
        self._build_indices()

        # Optionaler Konsistenz-Check
        if self.pre_feats is not None:
            expected = len(self.train_pairs) + len(self.val_pairs) + len(self.test_pairs)
            assert len(self.pre_feats) == expected, \
                f"pre_feats {len(self.pre_feats)} != erwartet {expected}"

        print(f"✓ Dataset ready | Train {len(self.train_pairs)} | "
              f"Val {len(self.val_pairs)} | Test {len(self.test_pairs)}")

    # -------------------------------------------------- #
    #  Daten laden
    # -------------------------------------------------- #
    def _load_raw_data(self, esf_path, normal_path):
        with open(esf_path, "rb") as f:
            self.esf = orjson.loads(f.read())
        with open(normal_path, "rb") as f:
            self.norm = orjson.loads(f.read())

    def _init_cv(self, cv_path, fold):
        with open(cv_path, "r", encoding="utf-8") as f:
            cv = json.load(f)[fold]
        self.train_raw, self.val_raw, self.test_raw = (
            cv["train"], cv["val"], cv["test"]
        )

    def _load_precomputed_feats(self, path, fold):
        import gzip
        self.pre_feats = None

        if path is None:
            return

        # 1) Datei lesen (gz oder plain)
        try:
            if str(path).lower().endswith(".gz"):
                with gzip.open(path, "rb") as f:
                    data = orjson.loads(f.read())
            else:
                with open(path, "rb") as f:
                    data = orjson.loads(f.read())
        except orjson.JSONDecodeError as e:
            raise RuntimeError(
                f"Fehler beim Lesen von '{path}'. "
                f"Ist die Datei gz-komprimiert? → {e}"
            )

        if fold not in data:
            raise ValueError(f"Fold '{fold}' fehlt in {path}")

        # 2) Ausgerichtete Listen (gleiche Länge wie cv6_info.json)
        fld = data[fold]
        tr_aligned = fld.get("train", [])
        va_aligned = fld.get("val", [])
        te_aligned = fld.get("test", [])

        # 3) Masken exakt wie deine Filterlogik
        def _pair_ok(p):
            cls_r, inst_r, perc_r, idx_r = self._parse_key(p["esf_ref"])
            cls_s, inst_s, perc_s, idx_s = self._parse_key(p["esf_scan"])
            A = self._get_combined(cls_r, inst_r, perc_r, idx_r)
            B = self._get_combined(cls_s, inst_s, perc_s, idx_s)
            return (A is not None and B is not None and
                    A.shape == (11, 64) and B.shape == (11, 64))

        mask_tr = [_pair_ok(p) for p in self.train_raw]
        mask_va = [_pair_ok(p) for p in self.val_raw]
        mask_te = [_pair_ok(p) for p in self.test_raw]

        # 4) Platzhalter behandeln (None/kaputt -> 44 Nullen) und auf gekeepte Menge komprimieren
        def _compress(aligned_list, mask):
            out = []
            for keep, feats in zip(mask, aligned_list):
                if not keep:
                    continue
                if feats is None or not (isinstance(feats, list) and len(feats) == 44):
                    out.append([0.0] * 44)
                else:
                    out.append(feats)
            return np.asarray(out, dtype=np.float32)

        tr = _compress(tr_aligned, mask_tr)
        va = _compress(va_aligned, mask_va)
        te = _compress(te_aligned, mask_te)

        # 5) Finaler Stapel in der Reihenfolge train+val+test
        self.pre_feats = np.concatenate([tr, va, te], axis=0)

    # -------------------------------------------------- #
    #  Hilfs-Funktionen
    # -------------------------------------------------- #
    @staticmethod
    def _parse_key(key):
        p = key.split("_")
        return "_".join(p[:-3]), p[-3], p[-2], int(p[-1])

    def _get_vec(self, store, cls, inst, perc, idx, exp_len):
        try:
            vec = store[cls][str(inst)][str(perc)][idx]
            arr = np.asarray(vec, np.float32)
            return arr if arr.size == exp_len else None
        except Exception:
            return None

    def _get_combined(self, cls, inst, perc, idx):
        e = self._get_vec(self.esf, cls, inst, perc, idx, 640)
        n = self._get_vec(self.norm, cls, inst, perc, idx, 64)
        if e is None or n is None:
            return None
        return np.concatenate([e.reshape(10, 64), n.reshape(1, 64)], 0)  # (11,64)

    @staticmethod
    def _cos_sim(v1, v2):
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-8 or n2 < 1e-8:
            return 0.0
        return float(np.dot(v1, v2) / (n1 * n2))

    # 11×4-Feature-Block
    def _row_metrics(self, v1, v2):
        cos_local = 1.0 - self._cos_sim(v1, v2)
        v1n = v1 / (np.linalg.norm(v1) + 1e-8)
        v2n = v2 / (np.linalg.norm(v2) + 1e-8)
        cos_global = 1.0 - self._cos_sim(v1n, v2n)
        emd_local = wasserstein_distance(v1, v2) / 64.0
        emd_global = wasserstein_distance(v1n, v2n) / 64.0
        return cos_local, cos_global, emd_local, emd_global

    # Augmentierungen ---------------------------------------------------- #
    @staticmethod
    def _aug_norm_rot(mat8x8):
        s = np.random.randint(1, 8)
        return np.roll(mat8x8, s, axis=0).reshape(1, 64)

    @staticmethod
    def _aug_noise(arr):
        return arr + np.random.normal(0, 0.01, arr.shape)

    # -------------------------------------------------- #
    #  Paar-Filterung & Indizes
    # -------------------------------------------------- #
    def _pair_ok(self, p):
        cls_r, inst_r, perc_r, idx_r = self._parse_key(p["esf_ref"])
        cls_s, inst_s, perc_s, idx_s = self._parse_key(p["esf_scan"])
        A = self._get_combined(cls_r, inst_r, perc_r, idx_r)
        B = self._get_combined(cls_s, inst_s, perc_s, idx_s)
        return (A is not None and B is not None and
                A.shape == (11, 64) and B.shape == (11, 64))

    def _filter_pairs(self, raw_pairs):
        ok = []
        for p in raw_pairs:
            if self._pair_ok(p):
                ok.append(p)
        return ok

    def _build_indices(self):
        self.train_pairs = self._filter_pairs(self.train_raw)
        self.val_pairs = self._filter_pairs(self.val_raw)
        self.test_pairs = self._filter_pairs(self.test_raw)
        self.all_pairs = self.train_pairs + self.val_pairs + self.test_pairs
        self.train_idx = np.arange(0, len(self.train_pairs))
        self.val_idx = np.arange(len(self.train_pairs),
                                 len(self.train_pairs) + len(self.val_pairs))
        self.test_idx = np.arange(len(self.all_pairs) - len(self.test_pairs),
                                  len(self.all_pairs))

    # -------------------------------------------------- #
    #  Haupt-Preprocessing-Pipeline
    # -------------------------------------------------- #
    def _do_preprocessing(self, i):
        pair = self.all_pairs[i]
        cls_r, inst_r, perc_r, idx_r = self._parse_key(pair["esf_ref"])
        cls_s, inst_s, perc_s, idx_s = self._parse_key(pair["esf_scan"])

        y = pair["label"]
        if y == 0:
            return None  # raus damit
        y = 0 if y == 2 else 1
        label = torch.tensor(y, dtype=torch.long)

        A = self._get_combined(cls_r, inst_r, perc_r, idx_r)  # (11,64)
        B = self._get_combined(cls_s, inst_s, perc_s, idx_s)
        if (A is None or B is None or
                A.shape != (11, 64) or B.shape != (11, 64)):
            return None

        # -------- Augmentierung --------
        diff = np.abs(A - B)  # (11,64)
        if np.random.rand() < .5:  # rotate nur Normal-Hist
            diff[-1:] = self._aug_norm_rot(diff[-1].reshape(8, 8))
        if np.random.rand() < .5:  # Noise auf allem
            diff = self._aug_noise(diff)

        # -------- Kanäle --------
        ch1 = diff / (np.max(np.abs(diff), axis=1, keepdims=True) + 1e-8)
        ch2 = diff / (np.max(np.abs(diff)) + 1e-8)
        grad = np.abs(np.gradient(ch1, axis=1))
        grad2 = np.abs(np.gradient(ch2, axis=1))
        stack = np.stack([ch1, ch2, grad, grad2]).astype(np.float32)  # (4,11,64)
        ch_tensor = torch.from_numpy(stack)

        # -------- Extra-Features --------
        if self.use_metrics:  # online berechnen
            feats = []
            for r in range(11):
                feats.extend(self._row_metrics(A[r], B[r]))
            if np.random.rand() < .3:  # leichtes Rauschen
                feats = np.asarray(feats) + np.random.normal(0, .01, len(feats))
            extra_tensor = torch.tensor(feats, dtype=torch.float32)
        else:  # vorgefertigte nutzen
            if self.pre_feats is None:
                extra_tensor = torch.tensor([])
            else:
                extra_tensor = torch.tensor(self.pre_feats[i], dtype=torch.float32)

        if label == 2:
            return (ch_tensor, extra_tensor, torch.tensor(0, dtype=torch.long))

        return ch_tensor, extra_tensor, label

    # -------------------------------------------------- #
    #  PyTorch-Dataset-API
    # -------------------------------------------------- #
    def __len__(self):
        return len(self.all_pairs)

    def __getitem__(self, idx):
        # Robust: springt weiter, bis ein gültiges Sample kommt
        start = idx
        while True:
            res = self._do_preprocessing(idx)
            if res is not None:
                return res
            idx = (idx + 1) % len(self.all_pairs)
            if idx == start:
                raise RuntimeError("Kein gültiges Sample im Dataset gefunden.")

    # -------------------------------------------------- #
    #  Loader-Helper
    # -------------------------------------------------- #
    def get_loaders(self, batch_size=32, batch_size_val=None, num_workers=0):
        """
        Liefert (train_loader, val_loader, test_loader).
        """
        train_idx = getattr(self, "train_indices", getattr(self, "train_idx"))
        val_idx = getattr(self, "val_indices", getattr(self, "val_idx"))
        test_idx = getattr(self, "test_indices", getattr(self, "test_idx"))

        val_bs = batch_size_val if batch_size_val is not None else batch_size
        pw = bool(num_workers > 0)  # nur True erlauben, wenn >0 (Windows!)

        train_loader = DataLoader(
            self,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(train_idx),
            pin_memory=True,
            num_workers=num_workers,
            persistent_workers=pw,
            drop_last=False,
        )
        val_loader = DataLoader(
            self,
            batch_size=val_bs,
            sampler=SubsetRandomSampler(val_idx),
            pin_memory=True,
            num_workers=num_workers,
            persistent_workers=pw,
            drop_last=False,
        )
        test_loader = DataLoader(
            self,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(test_idx),
            pin_memory=True,
            num_workers=num_workers,
            persistent_workers=pw,
            drop_last=False,
        )
        return train_loader, val_loader, test_loader


class ESFRefPairDatasetChannels_4_Xfeat(Dataset):
    # -------------------------------------------------- #
    #  Initialisierung
    # -------------------------------------------------- #
    def __init__(self, esf_path, normal_path, cv_path,
                 fold: str, use_metrics=False, extra_feats_path=None):
        self.use_metrics = use_metrics            # online-Features berechnen?
        self._feat_mode = None                    # "memmap" | "json" | None
        self._load_raw_data(esf_path, normal_path)
        self._init_cv(cv_path, fold)
        self._load_precomputed_feats(extra_feats_path, fold)
        self._build_indices()

        # Konsistenz-Checks nach dem Bauen der Indizes
        if self._feat_mode == "memmap":
            assert self._feat_tr.shape[0] == len(self.train_pairs), \
                f"train .npy Zeilen != gefilterte Train-Paare ({self._feat_tr.shape[0]} != {len(self.train_pairs)})"
            assert self._feat_va.shape[0] == len(self.val_pairs), \
                f"val .npy Zeilen != gefilterte Val-Paare ({self._feat_va.shape[0]} != {len(self.val_pairs)})"
            assert self._feat_te.shape[0] == len(self.test_pairs), \
                f"test .npy Zeilen != gefilterte Test-Paare ({self._feat_te.shape[0]} != {len(self.test_pairs)})"
        elif self.pre_feats is not None:
            expected = len(self.train_pairs) + len(self.val_pairs) + len(self.test_pairs)
            assert len(self.pre_feats) == expected, \
                f"pre_feats {len(self.pre_feats)} != erwartet {expected}"

        print(f"✓ Dataset ready | Train {len(self.train_pairs)} | "
              f"Val {len(self.val_pairs)} | Test {len(self.test_pairs)}")

    # -------------------------------------------------- #
    #  Daten laden
    # -------------------------------------------------- #
    def _load_raw_data(self, esf_path, normal_path):
        with open(esf_path,  "rb") as f: self.esf = orjson.loads(f.read())
        with open(normal_path, "rb") as f: self.norm = orjson.loads(f.read())

    def _init_cv(self, cv_path, fold):
        with open(cv_path, "r", encoding="utf-8") as f: cv = json.load(f)[fold]
        self.train_raw, self.val_raw, self.test_raw = (cv["train"], cv["val"], cv["test"])

    def _load_precomputed_feats(self, path, fold):
        """
        Unterstützt zwei Modi:
        - Ordner mit .npy pro Split (memmap; empfohlen):  <path>/<fold>_train.npy, _val.npy, _test.npy
        - Eine große JSON(.gz) im alten Format (lädt in RAM; nicht empfohlen bei >GB)
        """
        self.pre_feats = None
        if path is None:
            return

        if os.path.isdir(path):
            # ---- memmap-Modus ----
            tr_p = os.path.join(path, f"{fold}_train.npy")
            va_p = os.path.join(path, f"{fold}_val.npy")
            te_p = os.path.join(path, f"{fold}_test.npy")
            for p in (tr_p, va_p, te_p):
                if not os.path.exists(p):
                    raise FileNotFoundError(f"Fehlt: {p}")
            self._feat_tr = np.load(tr_p, mmap_mode="r")  # shape: (Ntr,44)
            self._feat_va = np.load(va_p, mmap_mode="r")  # shape: (Nva,44)
            self._feat_te = np.load(te_p, mmap_mode="r")  # shape: (Nte,44)
            self._feat_mode = "memmap"
            return

        # ---- Fallback: eine JSON/JSON.GZ laden (RAM) ----
        self._feat_mode = "json"
        if str(path).lower().endswith(".gz"):
            with gzip.open(path, "rb") as f:
                data = orjson.loads(f.read())
        else:
            with open(path, "rb") as f:
                data = orjson.loads(f.read())
        fld = data.get(fold, {})
        self.pre_feats = (fld.get("train", []) + fld.get("val", []) + fld.get("test", []))

    def _y_mapped(self, p):
        # 0 => verwerfen; 2 => 0; 1 => 1
        y = p["label"]
        if y == 0:  # leer raus
            return None
        return 0 if y == 2 else 1

    def _make_class_weights(self, alpha_neg=1.0):
        # counts nur über Train-Split
        n0 = n1 = 0
        for i in self.train_idx:
            y = self._y_mapped(self.all_pairs[i])
            if y is None:
                continue
            n0 += (y == 0)
            n1 += (y == 1)
        N = max(n0 + n1, 1)
        # inverse freq
        w0 = N / (2 * max(n0, 1))
        w1 = N / (2 * max(n1, 1))
        # Kostenprior: Negativ teurer (drückt FP1)
        w0 *= alpha_neg
        # auf Mittelwert 1 normalisieren (optional)
        s = w0 + w1
        w0, w1 = 2 * w0 / s, 2 * w1 / s
        return float(w0), float(w1)

    # -------------------------------------------------- #
    #  Hilfs-Funktionen
    # -------------------------------------------------- #
    @staticmethod
    def _parse_key(key):
        p = key.split("_");  return "_".join(p[:-3]), p[-3], p[-2], int(p[-1])

    def _get_vec(self, store, cls, inst, perc, idx, exp_len):
        try:
            vec = store[cls][str(inst)][str(perc)][idx]
            arr = np.asarray(vec, np.float32)
            return arr if arr.size == exp_len else None
        except Exception:
            return None

    def _get_combined(self, cls, inst, perc, idx):
        e = self._get_vec(self.esf,  cls, inst, perc, idx, 640)
        n = self._get_vec(self.norm, cls, inst, perc, idx,  64)
        if e is None or n is None: return None
        return np.concatenate([e.reshape(10, 64), n.reshape(1, 64)], 0)  # (11,64)

    # 11×4-Feature-Block (falls online berechnet wird)
    @staticmethod
    def _row_metrics(v1, v2):
        cos_local = 1 - _cos(v1, v2)
        v1n = v1 / (np.linalg.norm(v1) + 1e-8)
        v2n = v2 / (np.linalg.norm(v2) + 1e-8)
        cos_global = 1 - _cos(v1n, v2n)
        emd_local  = wasserstein_distance(v1, v2) / 64.0
        emd_global = wasserstein_distance(v1n, v2n) / 64.0
        return cos_local, cos_global, emd_local, emd_global

    # ---- Extra-Features holen (memmap/json) ----
    def _get_extra_feat(self, i):
        if self.use_metrics:
            return None  # wird on-the-fly gebaut

        if self._feat_mode == "memmap":
            n_tr = len(self.train_pairs)
            n_va = len(self.val_pairs)
            if i < n_tr:
                v = self._feat_tr[i]
            elif i < n_tr + n_va:
                v = self._feat_va[i - n_tr]
            else:
                v = self._feat_te[i - n_tr - n_va]
            return torch.tensor(v, dtype=torch.float32)

        if self.pre_feats is None:
            return torch.tensor([])

        return torch.tensor(self.pre_feats[i], dtype=torch.float32)

    # Augmentierungen ---------------------------------------------------- #
    @staticmethod
    def _aug_norm_rot(mat8x8):
        s = np.random.randint(1, 8)
        return np.roll(mat8x8, s, axis=0).reshape(1, 64)

    @staticmethod
    def _aug_noise(arr):
        return arr + np.random.normal(0, 0.01, arr.shape)

    # -------------------------------------------------- #
    #  Paar-Filterung & Indizes
    # -------------------------------------------------- #


    def _filter_pairs(self, raw_pairs):
        ok = []
        for p in raw_pairs:
            cls_r, inst_r, perc_r, idx_r = self._parse_key(p["esf_ref"])
            cls_s, inst_s, perc_s, idx_s = self._parse_key(p["esf_scan"])
            a = self._get_combined(cls_r, inst_r, perc_r, idx_r)
            b = self._get_combined(cls_s, inst_s, perc_s, idx_s)
            if a is not None and b is not None:
                ok.append(p)
        return ok

    def _build_indices(self):
        self.train_pairs = self._filter_pairs(self.train_raw)
        self.val_pairs   = self._filter_pairs(self.val_raw)
        self.test_pairs  = self._filter_pairs(self.test_raw)
        self.all_pairs   = self.train_pairs + self.val_pairs + self.test_pairs
        self.train_idx = np.arange(0, len(self.train_pairs))
        self.val_idx   = np.arange(len(self.train_pairs),
                                   len(self.train_pairs)+len(self.val_pairs))
        self.test_idx  = np.arange(len(self.all_pairs)-len(self.test_pairs),
                                   len(self.all_pairs))

    # -------------------------------------------------- #
    #  Haupt-Preprocessing-Pipeline
    # -------------------------------------------------- #
    def _do_preprocessing(self, i):
        pair = self.all_pairs[i]
        cls_r, inst_r, perc_r, idx_r = self._parse_key(pair["esf_ref"])
        cls_s, inst_s, perc_s, idx_s = self._parse_key(pair["esf_scan"])
        y = pair["label"]
        if y == 0:
            return None  # raus damit
        y = 0 if y == 2 else 1
        label = torch.tensor(y, dtype=torch.long)

        A = self._get_combined(cls_r, inst_r, perc_r, idx_r)  # (11,64)
        B = self._get_combined(cls_s, inst_s, perc_s, idx_s)
        if (A is None or B is None or A.shape != (11, 64) or B.shape != (11, 64)):
            return None

        # -------- Augmentierung --------
        diff = np.abs(A - B)                        # (11,64)
        if np.random.rand() < .5:                   # rotate nur Normal-Hist
            diff[-1:] = self._aug_norm_rot(diff[-1].reshape(8, 8))
        if np.random.rand() < .5:                   # Noise auf allem
            diff = self._aug_noise(diff)

        # -------- Kanäle --------
        ch1 = diff / (np.max(np.abs(diff), axis=1, keepdims=True)+1e-8)
        ch2 = diff / (np.max(np.abs(diff))+1e-8)
        grad = np.abs(np.gradient(ch1, axis=1))
        grad2 = np.abs(np.gradient(ch2, axis=1))
        stack = np.stack([ch1, ch2, grad, grad2]).astype(np.float32)  # (4,11,64)
        ch_tensor = torch.from_numpy(stack)

        # -------- Extra-Features --------
        if self.use_metrics:               # online berechnen
            feats = []
            for r in range(11):
                feats.extend(self._row_metrics(A[r], B[r]))
            if np.random.rand() < .3:
                feats = np.asarray(feats) + np.random.normal(0, .01, len(feats))
            extra_tensor = torch.tensor(feats, dtype=torch.float32)
        else:                              # vorgefertigte nutzen
            extra_tensor = self._get_extra_feat(i)
            if extra_tensor is None:
                extra_tensor = torch.tensor([])

        return ch_tensor, extra_tensor, label

    # -------------------------------------------------- #
    #  PyTorch-Dataset-API
    # -------------------------------------------------- #
    def __len__(self): return len(self.all_pairs)

    def __getitem__(self, idx):
        # Robust: springt weiter, bis ein gültiges Sample kommt
        start = idx
        while True:
            res = self._do_preprocessing(idx)
            if res is not None: return res
            idx = (idx + 1) % len(self.all_pairs)
            if idx == start:
                raise RuntimeError("Kein gültiges Sample im Dataset.")

    # -------------------------------------------------- #
    #  Loader-Helper
    # -------------------------------------------------- #


    def get_loaders(self, batch_size=32, batch_size_val=None, num_workers=0,
                    use_weighted_sampler=True, alpha_neg=1.3):
        train_idx = getattr(self, "train_idx")
        val_idx = getattr(self, "val_idx")
        test_idx = getattr(self, "test_idx")

        val_bs = batch_size_val if batch_size_val is not None else batch_size
        pw = bool(num_workers > 0)

        # Subsets bauen (wichtig für WeightedRandomSampler)
        ds_train = Subset(self, train_idx)
        ds_val = Subset(self, val_idx)
        ds_test = Subset(self, test_idx)

        if use_weighted_sampler:
            # 1) rohe Counts über train_idx (ohne Sampler)
            n0 = n1 = 0
            for i in train_idx:
                y = self._y_mapped(self.all_pairs[i])
                if y is None:
                    continue
                if y == 0:
                    n0 += 1
                else:
                    n1 += 1

            # 2) reine inverse Häufigkeit für den Sampler
            w_samp0 = 1.0 / max(n0, 1)
            w_samp1 = 1.0 / max(n1, 1)

            # 3) Sample-Gewichte je Beispiel
            w_samples = []
            for i in train_idx:
                y = self._y_mapped(self.all_pairs[i])
                if y is None:
                    w_samples.append(0.0)  # nie ziehen
                else:
                    w_samples.append(w_samp0 if y == 0 else w_samp1)

            sampler = WeightedRandomSampler(w_samples,
                                            num_samples=len(w_samples),
                                            replacement=True)

            train_loader = DataLoader(ds_train, batch_size=batch_size,
                                      sampler=sampler,
                                      pin_memory=True, num_workers=num_workers,
                                      persistent_workers=pw, drop_last=False)
        else:
            train_loader = DataLoader(ds_train, batch_size=batch_size,
                                      shuffle=True,
                                      pin_memory=True, num_workers=num_workers,
                                      persistent_workers=pw, drop_last=False)

        # Val/Test ohne Gewichtung
        val_loader = DataLoader(ds_val, batch_size=val_bs, shuffle=False,
                                pin_memory=True, num_workers=num_workers,
                                persistent_workers=pw, drop_last=False)
        test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False,
                                 pin_memory=True, num_workers=num_workers,
                                 persistent_workers=pw, drop_last=False)
        return train_loader, val_loader, test_loader


import os, json, gzip, gc
import numpy as np
import orjson
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torch.utils.data import Dataset
from scipy.stats import wasserstein_distance

# ... deine übrigen Imports (np, _cos etc) ...

class ESFRefPairDatasetChannels_4_Xfeat_S(Dataset):
    # -------------------------------------------------- #
    #  Initialisierung
    # -------------------------------------------------- #
    def __init__(self, esf_path, normal_path, cv_path,
                 fold: str, use_metrics=False, extra_feats_path=None,
                 max_negatives_train=500_000,
                 max_negatives_val=None,
                 max_negatives_test=None,
                 rng_seed=42):
        self.use_metrics = use_metrics
        self._feat_mode = None  # "memmap" | "json" | None
        self.rng = np.random.default_rng(rng_seed)

        self._load_raw_data(esf_path, normal_path)
        self._init_cv(cv_path, fold)
        self._load_precomputed_feats(extra_feats_path, fold)

        # 1) Pairs bauen
        self._build_indices()

        # 2) Counts VOR Shrink
        self._report_split_counts("BEFORE")

        # 3) Negatives (label==2) pro Split kappen
        self._apply_negative_cap(
            max_negatives_train=max_negatives_train,
            max_negatives_val=max_negatives_val,
            max_negatives_test=max_negatives_test
        )

        # 4) Datenspeicher auf benötigte Einträge schrumpfen
        self._shrink_hist_stores()
        self._shrink_extra_features()

        # 5) Indizes nach Shrink neu aufbauen
        self._rebuild_indices_after_shrink()

        # 6) Counts NACH Shrink
        self._report_split_counts("AFTER")

        # Konsistenz (angepasst für Memmap/JSON)
        if self._feat_mode == "memmap":
            # Wir nutzen Mapping-Listen; Shapes müssen NICHT mehr == sein
            assert hasattr(self, "_feat_tr_idx") and hasattr(self, "_feat_va_idx") and hasattr(self, "_feat_te_idx")
        elif self.pre_feats is not None:
            expected = len(self.train_pairs) + len(self.val_pairs) + len(self.test_pairs)
            assert len(self.pre_feats) == expected, f"pre_feats {len(self.pre_feats)} != erwartet {expected}"

        print(f"✓ Dataset ready | Train {len(self.train_pairs)} | "
              f"Val {len(self.val_pairs)} | Test {len(self.test_pairs)}")

    # -------------------------------------------------- #
    #  Daten laden
    # -------------------------------------------------- #
    def _load_raw_data(self, esf_path, normal_path):
        with open(esf_path,  "rb") as f: self.esf = orjson.loads(f.read())
        with open(normal_path, "rb") as f: self.norm = orjson.loads(f.read())

    # Augmentierungen ---------------------------------------------------- #
    @staticmethod
    def _aug_norm_rot(mat8x8):
        s = np.random.randint(1, 8)
        return np.roll(mat8x8, s, axis=0).reshape(1, 64)

    @staticmethod
    def _aug_noise(arr):
        return arr + np.random.normal(0, 0.01, arr.shape)

    def _init_cv(self, cv_path, fold):
        with open(cv_path, "r", encoding="utf-8") as f: cv = json.load(f)[fold]
        self.train_raw, self.val_raw, self.test_raw = (cv["train"], cv["val"], cv["test"])

    def _load_precomputed_feats(self, path, fold):
        self.pre_feats = None
        if path is None:
            return

        if os.path.isdir(path):
            # Memmap-Modus (pro Split eine Datei)
            tr_p = os.path.join(path, f"{fold}_train.npy")
            va_p = os.path.join(path, f"{fold}_val.npy")
            te_p = os.path.join(path, f"{fold}_test.npy")
            for p in (tr_p, va_p, te_p):
                if not os.path.exists(p):
                    raise FileNotFoundError(f"Fehlt: {p}")
            self._feat_tr = np.load(tr_p, mmap_mode="r")
            self._feat_va = np.load(va_p, mmap_mode="r")
            self._feat_te = np.load(te_p, mmap_mode="r")
            # Index-Mappings werden später gesetzt:
            self._feat_tr_idx = None
            self._feat_va_idx = None
            self._feat_te_idx = None
            self._feat_mode = "memmap"
            return

        # Fallback: eine JSON/JSON.GZ laden (RAM)
        self._feat_mode = "json"
        if str(path).lower().endswith(".gz"):
            with gzip.open(path, "rb") as f:
                data = orjson.loads(f.read())
        else:
            with open(path, "rb") as f:
                data = orjson.loads(f.read())
        fld = data.get(fold, {})
        self.pre_feats = (fld.get("train", []) + fld.get("val", []) + fld.get("test", []))
        # Hinweis: Reihenfolge = train + val + test

    # -------------------------------------------------- #
    #  Utils
    # -------------------------------------------------- #
    def _parse_key(self, key):
        p = key.split("_");  return "_".join(p[:-3]), p[-3], p[-2], int(p[-1])

    def _get_vec(self, store, cls, inst, perc, idx, exp_len):
        try:
            vec = store[cls][str(inst)][str(perc)][idx]
            arr = np.asarray(vec, np.float32)
            return arr if arr.size == exp_len else None
        except Exception:
            return None

    def _get_combined(self, cls, inst, perc, idx):
        e = self._get_vec(self.esf,  cls, inst, perc, idx, 640)
        n = self._get_vec(self.norm, cls, inst, perc, idx,  64)
        if e is None or n is None: return None
        return np.concatenate([e.reshape(10, 64), n.reshape(1, 64)], 0)  # (11,64)

    @staticmethod
    def _row_metrics(v1, v2):
        cos_local = 1 - _cos(v1, v2)
        v1n = v1 / (np.linalg.norm(v1) + 1e-8)
        v2n = v2 / (np.linalg.norm(v2) + 1e-8)
        cos_global = 1 - _cos(v1n, v2n)
        emd_local  = wasserstein_distance(v1, v2) / 64.0
        emd_global = wasserstein_distance(v1n, v2n) / 64.0
        return cos_local, cos_global, emd_local, emd_global

    # -------------------------------------------------- #
    #  Paar-Filterung & Indizes
    # -------------------------------------------------- #
    def _filter_pairs(self, raw_pairs):
        ok = []
        for p in raw_pairs:
            # Label 0 wird komplett verworfen
            if p["label"] == 0:
                continue
            cls_r, inst_r, perc_r, idx_r = self._parse_key(p["esf_ref"])
            cls_s, inst_s, perc_s, idx_s = self._parse_key(p["esf_scan"])
            a = self._get_combined(cls_r, inst_r, perc_r, idx_r)
            b = self._get_combined(cls_s, inst_s, perc_s, idx_s)
            if a is not None and b is not None:
                ok.append(p)
        return ok

    def _y_mapped(self, p):
        """
        0 => verwerfen; 2 => 0; 1 => 1
        Hinweis: label==0 sollte nach dem Filtern nicht mehr vorkommen.
        Wir lassen die Abfrage trotzdem drin für Robustheit.
        """
        y = p["label"]
        if y == 0:
            return None
        return 0 if y == 2 else 1

    def _build_indices(self):
        self.train_pairs = self._filter_pairs(self.train_raw)
        self.val_pairs   = self._filter_pairs(self.val_raw)
        self.test_pairs  = self._filter_pairs(self.test_raw)

        # Ursprungsindex innerhalb des gefilterten Split-Blocks merken
        for j, p in enumerate(self.train_pairs):
            p["_orig_idx"] = j
        for j, p in enumerate(self.val_pairs):
            p["_orig_idx"] = j
        for j, p in enumerate(self.test_pairs):
            p["_orig_idx"] = j


        self.all_pairs   = self.train_pairs + self.val_pairs + self.test_pairs

        # Für spätere Shrinks merken wir uns die Offsets
        self._train_off = (0, len(self.train_pairs))
        self._val_off   = (len(self.train_pairs), len(self.train_pairs)+len(self.val_pairs))
        self._test_off  = (len(self.train_pairs)+len(self.val_pairs), len(self.all_pairs))

        self._rebuild_indices_after_shrink()

    def _rebuild_indices_after_shrink(self):
        self.all_pairs = self.train_pairs + self.val_pairs + self.test_pairs
        self.train_idx = np.arange(0, len(self.train_pairs))
        self.val_idx   = np.arange(len(self.train_pairs),
                                   len(self.train_pairs)+len(self.val_pairs))
        self.test_idx  = np.arange(len(self.all_pairs)-len(self.test_pairs),
                                   len(self.all_pairs))

    # -------------------------------------------------- #
    #  Reporting
    # -------------------------------------------------- #
    def _report_split_counts(self, tag):
        def counts(pairs):
            n1 = sum(1 for p in pairs if p["label"] == 1)
            n2 = sum(1 for p in pairs if p["label"] == 2)
            n0 = sum(1 for p in pairs if p["label"] == 0)
            return len(pairs), n1, n2, n0

        t_all, t1, t2, t0 = counts(self.train_pairs)
        v_all, v1, v2, v0 = counts(self.val_pairs)
        s_all, s1, s2, s0 = counts(self.test_pairs)

        print(f"[{tag}] COUNTS")
        print(f"  Train: total={t_all} | label1={t1} | label2={t2} | label0={t0}")
        print(f"  Val  : total={v_all} | label1={v1} | label2={v2} | label0={v0}")
        print(f"  Test : total={s_all} | label1={s1} | label2={s2} | label0={s0}")

    # -------------------------------------------------- #
    #  Shrinks
    # -------------------------------------------------- #
    def _apply_negative_cap(self, max_negatives_train=500_000,
                            max_negatives_val=None,
                            max_negatives_test=None):
        """
        Behalte alle label==1. Kappe label==2 pro Split.
        label==0 ist bereits entfernt.
        """
        def cap_split(pairs, cap):
            pos = [p for p in pairs if p["label"] == 1]
            neg = [p for p in pairs if p["label"] == 2]
            if cap is not None and len(neg) > cap:
                sel = set(self.rng.choice(len(neg), size=cap, replace=False).tolist())
                neg = [neg[i] for i in sel]
            return pos + neg

        self.train_pairs = cap_split(self.train_pairs, max_negatives_train)
        self.val_pairs   = cap_split(self.val_pairs,   max_negatives_val)
        self.test_pairs  = cap_split(self.test_pairs,  max_negatives_test)

        # Shuffle je Split, damit Mischung zufällig ist
        self.rng.shuffle(self.train_pairs)
        self.rng.shuffle(self.val_pairs)
        self.rng.shuffle(self.test_pairs)

    def _needed_hist_keys(self):
        """Sammelt alle (store, cls, inst, perc, idx) die wir wirklich brauchen."""
        need = set()
        for pairs in (self.train_pairs, self.val_pairs, self.test_pairs):
            for p in pairs:
                for k in ("esf_ref","esf_scan"):
                    cls, inst, perc, idx = self._parse_key(p[k])
                    need.add(("esf", cls, str(inst), str(perc), idx))
                    need.add(("norm", cls, str(inst), str(perc), idx))
        return need

    def _shrink_hist_stores(self):
        """Reduziert self.esf und self.norm auf die benötigten Einträge."""
        need = self._needed_hist_keys()

        def shrink_one(store_name, store):
            out = {}
            for tag, cls, inst, perc, idx in need:
                if tag != store_name: continue
                # sichere Navigation + Copy-on-write
                cls_d = out.setdefault(cls, {})
                inst_d = cls_d.setdefault(inst, {})
                perc_d = inst_d.setdefault(perc, {})
                try:
                    val = store[cls][inst][perc][idx]
                except Exception:
                    continue
                perc_d[idx] = val
            return out

        self.esf  = shrink_one("esf",  self.esf)
        self.norm = shrink_one("norm", self.norm)

        # GC, um RAM freizugeben
        gc.collect()

    def _shrink_extra_features(self):
        """
        Kürzt Extra-Features auf die selektierten Paare.
        - JSON: Liste neu aufbauen => alter RAM frei.
        - Memmap: Index-Mappings setzen (keine Kopie).
        """
        # Mapping: globale all_pairs-Positionen pro Split
        n_tr = len(self.train_pairs)
        n_va = len(self.val_pairs)
        # n_te = len(self.test_pairs)

        if self._feat_mode is None:
            return

        if self._feat_mode == "json":
            # Reihenfolge in pre_feats: train + val + test (Original!)
            # Wir müssen die neuen Indizes (nach Shrink) in diese Reihenfolge mappen.
            # Dafür bauen wir die neue Liste direkt aus den Splits.
            new_feats = []

            # alte Splits aus pre_feats rekonstruieren:
            # Wir kennen die ursprünglichen Längen über _train_off/_val_off/_test_off
            t0, t1 = self._train_off
            v0, v1 = self._val_off
            s0, s1 = self._test_off
            pre_tr = self.pre_feats[t0:t1]
            pre_va = self.pre_feats[v0:v1]
            pre_te = self.pre_feats[s0:s1]

            # Jetzt die gekappten Splits nach gleicher Reihenfolge neu aufbauen
            # Achtung: Wir haben self.train_pairs etc. bereits gekappt und geshuffled.
            # Wir benötigen die Positionen innerhalb des ursprünglichen Split-Blocks.
            def block_new_feats(pairs, block_feats):
                out = []
                for p in pairs:
                    # Ursprungsposition des Paares im Block via "_orig_idx" merken.
                    # Wenn nicht vorhanden, im Build eine anlegen (siehe unten).
                    idx_in_block = p["_orig_idx"]
                    out.append(block_feats[idx_in_block])
                return out

            new_feats.extend(block_new_feats(self.train_pairs, pre_tr))
            new_feats.extend(block_new_feats(self.val_pairs,   pre_va))
            new_feats.extend(block_new_feats(self.test_pairs,  pre_te))

            # Ersetzen + freigeben
            self.pre_feats = new_feats
            gc.collect()

        elif self._feat_mode == "memmap":
            # Wir legen nur Index-Mappings an, um aus den originalen Dateien zu lesen.
            # Dazu brauchen wir die Ursprungsposition je Paar im jeweiligen Split.
            self._feat_tr_idx = np.array([p["_orig_idx"] for p in self.train_pairs], dtype=np.int64)
            self._feat_va_idx = np.array([p["_orig_idx"] for p in self.val_pairs],   dtype=np.int64)
            self._feat_te_idx = np.array([p["_orig_idx"] for p in self.test_pairs],  dtype=np.int64)

    # -------------------------------------------------- #
    #  Haupt-Preprocessing-Pipeline
    # -------------------------------------------------- #
    def _do_preprocessing(self, i):
        pair = self.all_pairs[i]
        cls_r, inst_r, perc_r, idx_r = self._parse_key(pair["esf_ref"])
        cls_s, inst_s, perc_s, idx_s = self._parse_key(pair["esf_scan"])

        # label==0 kommt hier nicht mehr vor; 2 → 0
        y = pair["label"]
        y = 0 if y == 2 else 1
        label = torch.tensor(y, dtype=torch.long)

        A = self._get_combined(cls_r, inst_r, perc_r, idx_r)  # (11,64)
        B = self._get_combined(cls_s, inst_s, perc_s, idx_s)
        if (A is None or B is None or A.shape != (11, 64) or B.shape != (11, 64)):
            return None

        # -------- Augmentierung --------
        diff = np.abs(A - B)                        # (11,64)
        if np.random.rand() < .5:
            diff[-1:] = self._aug_norm_rot(diff[-1].reshape(8, 8))
        if np.random.rand() < .5:
            diff = self._aug_noise(diff)

        # -------- Kanäle --------
        ch1 = diff / (np.max(np.abs(diff), axis=1, keepdims=True)+1e-8)
        ch2 = diff / (np.max(np.abs(diff))+1e-8)
        grad = np.abs(np.gradient(ch1, axis=1))
        grad2 = np.abs(np.gradient(ch2, axis=1))
        stack = np.stack([ch1, ch2, grad, grad2]).astype(np.float32)  # (4,11,64)
        ch_tensor = torch.from_numpy(stack)

        # -------- Extra-Features --------
        if self.use_metrics:
            feats = []
            for r in range(11):
                feats.extend(self._row_metrics(A[r], B[r]))
            if np.random.rand() < .3:
                feats = np.asarray(feats) + np.random.normal(0, .01, len(feats))
            extra_tensor = torch.tensor(feats, dtype=torch.float32)
        else:
            extra_tensor = self._get_extra_feat(i)
            if extra_tensor is None:
                extra_tensor = torch.tensor([])

        return ch_tensor, extra_tensor, label

    # -------------------------------------------------- #
    #  Extra-Features holen (angepasst)
    # -------------------------------------------------- #
    def _get_extra_feat(self, i):
        if self.use_metrics:
            return None

        if self._feat_mode == "memmap":
            n_tr = len(self.train_pairs)
            n_va = len(self.val_pairs)
            if i < n_tr:
                # Mapping auf ursprüngliche Zeile
                j = int(self._feat_tr_idx[i])
                v = self._feat_tr[j]
            elif i < n_tr + n_va:
                j = int(self._feat_va_idx[i - n_tr])
                v = self._feat_va[j]
            else:
                j = int(self._feat_te_idx[i - n_tr - n_va])
                v = self._feat_te[j]
            return torch.tensor(v, dtype=torch.float32)

        if self.pre_feats is None:
            return torch.tensor([])

        return torch.tensor(self.pre_feats[i], dtype=torch.float32)

    # -------------------------------------------------- #
    #  PyTorch-Dataset-API
    # -------------------------------------------------- #
    def __len__(self): return len(self.all_pairs)

    def __getitem__(self, idx):
        start = idx
        while True:
            res = self._do_preprocessing(idx)
            if res is not None: return res
            idx = (idx + 1) % len(self.all_pairs)
            if idx == start:
                raise RuntimeError("Kein gültiges Sample im Dataset.")

    # -------------------------------------------------- #
    #  Loader-Helper (unverändert, aber: WeightedSampler eher AUS)
    # -------------------------------------------------- #
    def get_loaders(self, batch_size=32, batch_size_val=None, num_workers=0,
                    use_weighted_sampler=False, alpha_neg=1.0):
        train_idx = getattr(self, "train_idx")
        val_idx = getattr(self, "val_idx")
        test_idx = getattr(self, "test_idx")

        val_bs = batch_size_val if batch_size_val is not None else batch_size
        pw = bool(num_workers > 0)

        ds_train = Subset(self, train_idx)
        ds_val   = Subset(self, val_idx)
        ds_test  = Subset(self, test_idx)

        if use_weighted_sampler:
            # Achtung: Nach Kappung meist nicht nötig.
            n0 = n1 = 0
            for i in train_idx:
                y = self.all_pairs[i]["label"]
                y = 0 if y == 2 else 1
                if y == 0: n0 += 1
                else:      n1 += 1

            w_samp0 = 1.0 / max(n0, 1)
            w_samp1 = 1.0 / max(n1, 1)
            w_samples = []
            for i in train_idx:
                y = self.all_pairs[i]["label"]
                y = 0 if y == 2 else 1
                w_samples.append(w_samp0 if y == 0 else w_samp1)

            sampler = WeightedRandomSampler(w_samples,
                                            num_samples=len(w_samples),
                                            replacement=True)

            train_loader = DataLoader(ds_train, batch_size=batch_size,
                                      sampler=sampler,
                                      pin_memory=True, num_workers=num_workers,
                                      persistent_workers=pw, drop_last=False)
        else:
            train_loader = DataLoader(ds_train, batch_size=batch_size,
                                      shuffle=True,
                                      pin_memory=True, num_workers=num_workers,
                                      persistent_workers=pw, drop_last=False)

        val_loader = DataLoader(ds_val, batch_size=val_bs, shuffle=False,
                                pin_memory=True, num_workers=num_workers,
                                persistent_workers=pw, drop_last=False)
        test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False,
                                 pin_memory=True, num_workers=num_workers,
                                 persistent_workers=pw, drop_last=False)
        return train_loader, val_loader, test_loader



    ## neuen dTssets dür neue daten

    import os, json, gzip, gc
    import orjson, numpy as np, torch
    from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler

    from scipy.spatial.distance import cosine as _cos
    from scipy.stats import wasserstein_distance

    # ---------- kleine Helfer ----------
    def _parse_key(self, key):
        p = key.split("_");
        return "_".join(p[:-3]), p[-3], p[-2], int(p[-1])

    def _safe_get(self, store, cls, inst, perc, idx, exp_len):
        try:
            v = store[cls][str(inst)][str(perc)][idx]
            a = np.asarray(v, np.float32)
            return a if a.size == exp_len else None
        except Exception:
            return None

    def _row_metrics(self, v1, v2):
        cos_local = 1 - _cos(v1, v2)
        v1n = v1 / (np.linalg.norm(v1) + 1e-8)
        v2n = v2 / (np.linalg.norm(v2) + 1e-8)
        cos_global = 1 - _cos(v1n, v2n)
        emd_local = wasserstein_distance(v1, v2) / 64.0
        emd_global = wasserstein_distance(v1n, v2n) / 64.0
        return cos_local, cos_global, emd_local, emd_global

    # ---------- Dataset: direkter 704d-Diff ----------


import os, json, gzip, gc
import orjson, numpy as np, torch
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from scipy.spatial.distance import cosine as _cos
from scipy.stats import wasserstein_distance

class ESFRefPairDatasetMLP704Fast(Dataset):
    """
    Liefert:
      x704 : abs(esf_ref - esf_scan) [640]  ||  aug(abs(norm_ref - norm_scan)) [64]  -> (704,)
      xext : optionale 44d-Extras (memmap/json oder on-the-fly)
      y    : 0 (anderes Objekt; ehem. label=2) / 1 (vorhanden)
    label==0 wird vollständig verworfen.
    """

    # ----------------------------- Init ----------------------------- #
    def __init__(self,
                 esf_path, normal_path, cv_path, fold: str,
                 use_metrics=False, extra_feats_path=None, grid_feats_path=None,
                 max_negatives_train=500_000, max_negatives_val=None, max_negatives_test=None,
                 rng_seed=42, aug_rotate_norm=True, aug_noise_std=0.01,
                 drop_pos_perc_values=None,          # z.B. {"100|30","100|35","100|40","100|45"}
                 drop_pos_where = "either", # "ref" | "scan" | "either"
                 debug = True, xext_policy="all"  # <<< NEU: "as_is" | "extra" | "grid" | "all"
                 ):
        self.xext_policy = str(xext_policy).lower()
        self.use_metrics     = bool(use_metrics)
        self._feat_mode      = None     # "memmap" | "json" | None
        self.rng             = np.random.default_rng(rng_seed)
        self.aug_rotate_norm = bool(aug_rotate_norm)
        self.aug_noise_std   = float(aug_noise_std)
        self.drop_pos_perc_values = set(drop_pos_perc_values or [])
        self.drop_pos_where = str(drop_pos_where).lower()
        self.debug = bool(debug)
        self.drop_pos_perc_values = set(drop_pos_perc_values or [])
        self._ban_perc = {self._canon_perc(x) for x in self.drop_pos_perc_values}
        if self.drop_pos_where not in {"ref", "scan", "either"}:
            raise ValueError("drop_pos_where must be one of {'ref','scan','either'}")

        # Rohdaten laden
        with open(esf_path,   "rb") as f: self.esf  = orjson.loads(f.read())
        with open(normal_path,"rb") as f: self.norm = orjson.loads(f.read())

        # CV laden
        with open(cv_path, "r", encoding="utf-8") as f:
            cv = json.load(f)[fold]
        self.train_raw, self.val_raw, self.test_raw = cv["train"], cv["val"], cv["test"]

        # Extra-Features (memmap/json) optional
        self._load_precomputed_feats(extra_feats_path, grid_feats_path, fold)

        # 1) Paare bauen + _orig_idx setzen
        self._build_indices()

        # 2) Vorher-Counts
        self._report_split_counts("BEFORE")

        # 3) Negatives (label==2) pro Split kappen
        self._apply_negative_cap(max_negatives_train, max_negatives_val, max_negatives_test)

        # 4) Stores auf benötigte Keys schrumpfen
        self._shrink_hist_stores()

        # 5) Extra-Features schrumpfen (robust; Memmap: OOB filtern)
        self._shrink_extra_features()

        # 6) Indizes nach Shrink neu
        self._rebuild_indices_after_shrink()

        # 7) Nachher-Counts
        self._report_split_counts("AFTER")

        # Konsistenz (robuster Modus: nur prüfen, ob Mappings existieren)
        if self._feat_mode == "memmap":
            assert hasattr(self, "_feat_tr_idx") and hasattr(self, "_feat_va_idx") and hasattr(self, "_feat_te_idx")
        elif getattr(self, "pre_feats", None) is not None:
            expected = len(self.train_pairs) + len(self.val_pairs) + len(self.test_pairs)
            assert len(self.pre_feats) == expected, f"pre_feats {len(self.pre_feats)} != erwartet {expected}"

        print(f"✓ Dataset ready | Train {len(self.train_pairs)} | Val {len(self.val_pairs)} | Test {len(self.test_pairs)}")

    # ----------------------------- Helpers ----------------------------- #
    def _y_mapped(self, p):
        # 0 => verwerfen; 2 => 0; 1 => 1
        y = p["label"]
        if y == 0:
            return None
        return 0 if y == 2 else 1

    @staticmethod
    def _parse_key(key: str):
        p = key.split("_")
        return "_".join(p[:-3]), p[-3], p[-2], int(p[-1])

    @staticmethod
    def _safe_get(store, cls, inst, perc, idx, exp_len):
        try:
            v = store[cls][str(inst)][str(perc)][idx]
            a = np.asarray(v, np.float32)
            return a if a.size == exp_len else None
        except Exception:
            return None

    @staticmethod
    def _row_metrics(v1, v2):
        cos_local = 1 - _cos(v1, v2)
        v1n = v1 / (np.linalg.norm(v1) + 1e-8)
        v2n = v2 / (np.linalg.norm(v2) + 1e-8)
        cos_global = 1 - _cos(v1n, v2n)
        emd_local  = wasserstein_distance(v1, v2) / 64.0
        emd_global = wasserstein_distance(v1n, v2n) / 64.0
        # robust
        if not np.isfinite(cos_local):  cos_local = 0.0
        if not np.isfinite(cos_global): cos_global = 0.0
        if not np.isfinite(emd_local):  emd_local  = 1.0
        if not np.isfinite(emd_global): emd_global = 1.0
        return cos_local, cos_global, emd_local, emd_global

    def _safe_load_grid(self, path):
        try:
            # bevorzugt: echtes float-Array, memmapfähig
            arr = np.load(path, mmap_mode="r")
            return arr, True  # memmap_ok
        except ValueError as e:
            if "allow_pickle=False" in str(e):
                # Fallback: Pickle zulassen, aber OHNE memmap
                arr = np.load(path, allow_pickle=True)
                # falls object-Array: stapeln + casten
                if arr.dtype == object:
                    arr = np.stack(arr, axis=0).astype(np.float32, copy=False)
                return arr, False
            raise

    def _load_precomputed_feats(self, path, grid_path, fold):
        self.pre_feats = None
        self._feat_mode = None
        # GRID defaults
        self._grid_mode = None
        self._gfeat_tr = self._gfeat_va = self._gfeat_te = None
        self._gfeat_tr_idx = self._gfeat_va_idx = self._gfeat_te_idx = None
        self._grid_dim = 27  # wird unten ggf. aus Datei gelesen
        # --- 44er Extras ---
        if path is not None:
            if os.path.isdir(path):
                tr_p = os.path.join(path, f"{fold}_train.npy")
                va_p = os.path.join(path, f"{fold}_val.npy")
                te_p = os.path.join(path, f"{fold}_test.npy")
                for pth in (tr_p, va_p, te_p):
                    if not os.path.exists(pth):
                        raise FileNotFoundError(f"Fehlt: {pth}")
                self._feat_tr = np.load(tr_p, mmap_mode="r")
                self._feat_va = np.load(va_p, mmap_mode="r")
                self._feat_te = np.load(te_p, mmap_mode="r")
                self._feat_tr_idx = self._feat_va_idx = self._feat_te_idx = None
                self._feat_mode = "memmap"
            else:
                self._feat_mode = "json"
                if str(path).lower().endswith(".gz"):
                    import gzip
                    with gzip.open(path, "rb") as f:
                        data = orjson.loads(f.read())
                else:
                    with open(path, "rb") as f:
                        data = orjson.loads(f.read())
                fld = data.get(fold, {})
                self.pre_feats = fld.get("train", []) + fld.get("val", []) + fld.get("test", [])

        # --- 27er GRID ---
        if grid_path is not None:
            if not os.path.isdir(grid_path):
                raise ValueError(f"grid_feats_path muss ein Ordner sein: {grid_path}")
            gtr = os.path.join(grid_path, f"{fold}_train.npy")
            gva = os.path.join(grid_path, f"{fold}_val.npy")
            gte = os.path.join(grid_path, f"{fold}_test.npy")
            for pth in (gtr, gva, gte):
                if not os.path.exists(pth):
                    raise FileNotFoundError(f"Grid-Features fehlen: {pth}")

            self._gfeat_tr, tr_memmap = self._safe_load_grid(gtr)
            self._gfeat_va, va_memmap = self._safe_load_grid(gva)
            self._gfeat_te, te_memmap = self._safe_load_grid(gte)

            if self._gfeat_tr.ndim == 2:
                self._grid_dim = int(self._gfeat_tr.shape[1])
            self._grid_mode = "memmap"

    # ----------------------------- Build indices ----------------------------- #
    # --- Neu: Hilfsfunktion ---
    @staticmethod
    def _canon_perc(v):
        s = str(v).strip()
        # nimm den rechten Teil, falls "100|30" etc.
        if "|" in s:
            s = s.split("|")[-1].strip()
        return s

    def _is_banned_positive(self, p):
        # nur positive Paare (label==1) prüfen
        if p.get("label") != 1 or not self._ban_perc:
            return False

        cls_r, inst_r, perc_r, idx_r = self._parse_key(p["esf_ref"])
        cls_s, inst_s, perc_s, idx_s = self._parse_key(p["esf_scan"])

        # kanonisieren (z.B. "100|30" -> "30")
        pr = self._canon_perc(perc_r)
        ps = self._canon_perc(perc_s)

        ref_bad = (self.drop_pos_where in {"ref", "either"}) and (pr in self._ban_perc)
        scan_bad = (self.drop_pos_where in {"scan", "either"}) and (ps in self._ban_perc)
        banned = ref_bad or scan_bad

        if banned and self.debug:
            who = []
            if ref_bad:  who.append("ref")
            if scan_bad: who.append("scan")
            print(f"[DROP POS] perc_ref={perc_r}({pr}) perc_scan={perc_s}({ps}) -> via {','.join(who)}")

        return banned

    def _keep_pair(self, p):
        if p["label"] == 0:
            return False
        # <<< NEU: positives per 'perc' filtern
        if self._is_banned_positive(p):
            return False

        cls_r, inst_r, perc_r, idx_r = self._parse_key(p["esf_ref"])
        cls_s, inst_s, perc_s, idx_s = self._parse_key(p["esf_scan"])
        e_r = self._safe_get(self.esf, cls_r, inst_r, perc_r, idx_r, 640)
        e_s = self._safe_get(self.esf, cls_s, inst_s, perc_s, idx_s, 640)
        n_r = self._safe_get(self.norm, cls_r, inst_r, perc_r, idx_r, 64)
        n_s = self._safe_get(self.norm, cls_s, inst_s, perc_s, idx_s, 64)
        return (e_r is not None) and (e_s is not None) and (n_r is not None) and (n_s is not None)


    def _build_indices(self):
        self.train_pairs = []
        for i, p in enumerate(self.train_raw):
            if self._keep_pair(p):
                q = dict(p); q["_orig_idx"] = i
                self.train_pairs.append(q)

        self.val_pairs = []
        for i, p in enumerate(self.val_raw):
            if self._keep_pair(p):
                q = dict(p); q["_orig_idx"] = i
                self.val_pairs.append(q)

        self.test_pairs = []
        for i, p in enumerate(self.test_raw):
            if self._keep_pair(p):
                q = dict(p); q["_orig_idx"] = i
                self.test_pairs.append(q)

        self.all_pairs  = self.train_pairs + self.val_pairs + self.test_pairs
        # Offsets im originalen pre_feats-Layout (train + val + test)
        self._train_off = (0, len(self.train_pairs))
        self._val_off   = (len(self.train_pairs), len(self.train_pairs) + len(self.val_pairs))
        self._test_off  = (len(self.train_pairs) + len(self.val_pairs), len(self.all_pairs))
        self._rebuild_indices_after_shrink()

    def _rebuild_indices_after_shrink(self):
        self.all_pairs = self.train_pairs + self.val_pairs + self.test_pairs
        self.train_idx = np.arange(0, len(self.train_pairs))
        self.val_idx   = np.arange(len(self.train_pairs), len(self.train_pairs) + len(self.val_pairs))
        self.test_idx  = np.arange(len(self.all_pairs) - len(self.test_pairs), len(self.all_pairs))

    # ----------------------------- Reporting ----------------------------- #
    def _report_split_counts(self, tag):
        def counts(pairs):
            n1 = sum(1 for p in pairs if p["label"] == 1)
            n2 = sum(1 for p in pairs if p["label"] == 2)
            n0 = sum(1 for p in pairs if p["label"] == 0)
            return len(pairs), n1, n2, n0

        t_all, t1, t2, t0 = counts(self.train_pairs)
        v_all, v1, v2, v0 = counts(self.val_pairs)
        s_all, s1, s2, s0 = counts(self.test_pairs)

        print(f"[{tag}] COUNTS")
        print(f"  Train: total={t_all} | label1={t1} | label2={t2} | label0={t0}")
        print(f"  Val  : total={v_all} | label1={v1} | label2={v2} | label0={v0}")
        print(f"  Test : total={s_all} | label1={s1} | label2={s2} | label0={s0}")

    # ----------------------------- Shrinks ----------------------------- #
    def _apply_negative_cap(self, max_negatives_train=500_000, max_negatives_val=None, max_negatives_test=None):
        def cap_split(pairs, cap):
            pos = [p for p in pairs if p["label"] == 1]
            neg = [p for p in pairs if p["label"] == 2]
            if cap is not None and len(neg) > cap:
                sel = set(self.rng.choice(len(neg), size=cap, replace=False).tolist())
                neg = [neg[i] for i in sel]
            out = pos + neg
            self.rng.shuffle(out)
            return out

        self.train_pairs = cap_split(self.train_pairs, max_negatives_train)
        self.val_pairs   = cap_split(self.val_pairs,   max_negatives_val)
        self.test_pairs  = cap_split(self.test_pairs,  max_negatives_test)

    def _needed_hist_keys(self):
        need = set()
        for pairs in (self.train_pairs, self.val_pairs, self.test_pairs):
            for p in pairs:
                for k in ("esf_ref", "esf_scan"):
                    cls, inst, perc, idx = self._parse_key(p[k])
                    need.add(("esf",  cls, str(inst), str(perc), idx))
                    need.add(("norm", cls, str(inst), str(perc), idx))
        return need

    def _shrink_hist_stores(self):
        need = self._needed_hist_keys()

        def shrink_one(tag_name, store):
            out = {}
            for tag, cls, inst, perc, idx in need:
                if tag != tag_name: continue
                cls_d  = out.setdefault(cls, {})
                inst_d = cls_d.setdefault(inst, {})
                perc_d = inst_d.setdefault(perc, {})
                try:
                    val = store[cls][inst][perc][idx]
                except Exception:
                    continue
                perc_d[idx] = val
            return out

        self.esf  = shrink_one("esf",  self.esf)
        self.norm = shrink_one("norm", self.norm)
        gc.collect()

    def _shrink_extra_features(self):
        import gc
        if self._feat_mode == "json":
            # unverändert wie bisher ...
            t0, t1 = self._train_off;
            v0, v1 = self._val_off;
            s0, s1 = self._test_off
            pre_tr = self.pre_feats[t0:t1];
            pre_va = self.pre_feats[v0:v1];
            pre_te = self.pre_feats[s0:s1]

            def block_new_feats(pairs, block_feats):
                return [block_feats[p["_orig_idx"]] for p in pairs]

            new_feats = []
            new_feats.extend(block_new_feats(self.train_pairs, pre_tr))
            new_feats.extend(block_new_feats(self.val_pairs, pre_va))
            new_feats.extend(block_new_feats(self.test_pairs, pre_te))
            self.pre_feats = new_feats
            gc.collect()

        elif self._feat_mode == "memmap":
            # 44er-Extras: nur Indexmaps bauen
            self._feat_tr_idx = np.array([p["_orig_idx"] for p in self.train_pairs], dtype=np.int64)
            self._feat_va_idx = np.array([p["_orig_idx"] for p in self.val_pairs], dtype=np.int64)
            self._feat_te_idx = np.array([p["_orig_idx"] for p in self.test_pairs], dtype=np.int64)
            # OOB lassen wir zu; _get_extra gibt dann leeren Tensor/Nullen zurück

        # GRID-Memmaps: ebenfalls Indexmaps (falls vorhanden)
        if self._grid_mode == "memmap":
            self._gfeat_tr_idx = np.array([p["_orig_idx"] for p in self.train_pairs], dtype=np.int64)
            self._gfeat_va_idx = np.array([p["_orig_idx"] for p in self.val_pairs], dtype=np.int64)
            self._gfeat_te_idx = np.array([p["_orig_idx"] for p in self.test_pairs], dtype=np.int64)

    # ----------------------------- Core features ----------------------------- #
    def _diff704(self, e_r, e_s, n_r, n_s):
        de = np.abs(e_r - e_s).astype(np.float32)  # (640,)
        dn = np.abs(n_r - n_s).astype(np.float32)  # (64,)

        # Rotation nur auf NormalHist (8x8) – optional
        if self.aug_rotate_norm and (self.rng.random() < 0.5):
            dn = np.roll(dn.reshape(8, 8), self.rng.integers(1, 8), axis=0).reshape(64)

        x = np.concatenate([de, dn], 0)            # (704,)

        # Noise auf ALLE 704 – optional
        if (self.aug_noise_std > 0.0) and (self.rng.random() < 0.5):
            x = x + self.rng.normal(0.0, self.aug_noise_std, size=x.shape).astype(np.float32)

        return x

    def _get_extra(self, i):
        import torch, numpy as np

        # ------- 44D laden (wie bei dir, unverändert bis ext44/grid27 gebaut sind) -------
        # ext44
        ext44 = None
        if self.use_metrics:
            ext44 = np.empty((0,), dtype=np.float32)  # oder deine on-the-fly-Features
        elif self._feat_mode is None:
            ext44 = np.empty((0,), dtype=np.float32)
        elif self._feat_mode == "json":
            ext44 = np.asarray(self.pre_feats[i], dtype=np.float32)
        elif self._feat_mode == "memmap":
            n_tr = len(self.train_pairs);
            n_va = len(self.val_pairs)
            if i < n_tr:
                j = int(self._feat_tr_idx[i]);
                src = self._feat_tr
            elif i < n_tr + n_va:
                j = int(self._feat_va_idx[i - n_tr]);
                src = self._feat_va
            else:
                j = int(self._feat_te_idx[i - n_tr - n_va]);
                src = self._feat_te
            if 0 <= j < len(src):
                ext44 = np.asarray(src[j], dtype=np.float32)
            else:
                ext44 = np.empty((0,), dtype=np.float32)

        # grid27
        grid27 = None
        if self._grid_mode == "memmap":
            n_tr = len(self.train_pairs);
            n_va = len(self.val_pairs)
            if i < n_tr:
                j = int(self._gfeat_tr_idx[i]);
                src = self._gfeat_tr
            elif i < n_tr + n_va:
                j = int(self._gfeat_va_idx[i - n_tr]);
                src = self._gfeat_va
            else:
                j = int(self._gfeat_te_idx[i - n_tr - n_va]);
                src = self._gfeat_te
            if 0 <= j < len(src):
                grid27 = np.asarray(src[j], dtype=np.float32)
            else:
                grid27 = np.zeros((self._grid_dim,), dtype=np.float32)
        else:
            grid27 = np.empty((0,), dtype=np.float32)

        # ------- Fix-Längen Helpers -------
        def _fix_len(x: np.ndarray, D: int) -> np.ndarray:
            if x is None or x.size == 0:
                return np.zeros((D,), dtype=np.float32)
            x = x.reshape(-1).astype(np.float32, copy=False)
            if x.size == D:
                return x
            if x.size > D:
                return x[:D]
            out = np.zeros((D,), dtype=np.float32)
            out[:x.size] = x
            return out

        Dg = int(self._grid_dim)  # meist 27
        pol = getattr(self, "xext_policy", "as_is").lower()

        if pol == "extra":
            # immer 44D
            return torch.from_numpy(_fix_len(ext44, 44))

        if pol == "grid":
            # immer 27D
            return torch.from_numpy(_fix_len(grid27, Dg))

        if pol == "all":
            # immer 71D (44 + Dg)
            x44 = _fix_len(ext44, 44)
            x27 = _fix_len(grid27, Dg)
            return torch.from_numpy(np.concatenate([x44, x27], 0))

        # Fallback: altes Verhalten (kann mischen – NICHT empfohlen)
        if (ext44 is None or ext44.size == 0) and (grid27 is None or grid27.size == 0):
            return torch.tensor([])
        xcat = np.concatenate([ext44, grid27], 0).astype(np.float32, copy=False)
        return torch.from_numpy(xcat)
    # ----------------------------- Dataset API ----------------------------- #
    def __len__(self): return len(self.all_pairs)

    def __getitem__(self, idx):
        start = idx
        while True:
            p = self.all_pairs[idx]
            y = 0 if p["label"] == 2 else 1

            cls_r, inst_r, perc_r, idx_r = self._parse_key(p["esf_ref"])
            cls_s, inst_s, perc_s, idx_s = self._parse_key(p["esf_scan"])
            e_r = self._safe_get(self.esf, cls_r, inst_r, perc_r, idx_r, 640)
            e_s = self._safe_get(self.esf, cls_s, inst_s, perc_s, idx_s, 640)
            n_r = self._safe_get(self.norm, cls_r, inst_r, perc_r, idx_r, 64)
            n_s = self._safe_get(self.norm, cls_s, inst_s, perc_s, idx_s, 64)

            if (e_r is None) or (e_s is None) or (n_r is None) or (n_s is None):
                idx = (idx + 1) % len(self.all_pairs)
                if idx == start:
                    raise RuntimeError("Kein gültiges Sample im Dataset.")
                continue

            x704 = self._diff704(e_r, e_s, n_r, n_s)
            xext = self._get_extra(idx)
            return torch.from_numpy(x704), xext, torch.tensor(y, dtype=torch.long)#, torch.tensor(idx, dtype=torch.long)

    # ----------------------------- Loaders ----------------------------- #
    def get_loaders(self, batch_size=128, batch_size_val=None, num_workers=0,
                    use_weighted_sampler=False):
        val_bs = batch_size_val if batch_size_val is not None else batch_size

        ds_train = Subset(self, self.train_idx)
        ds_val   = Subset(self, self.val_idx)
        ds_test  = Subset(self, self.test_idx)

        pw = bool(num_workers > 0)
        if use_weighted_sampler:
            # inverse Häufigkeit auf train_idx (nach Kappung/Shuffle)
            n0 = sum(1 for i in self.train_idx if (0 if self.all_pairs[i]["label"] == 2 else 1) == 0)
            n1 = len(self.train_idx) - n0
            w0 = 1.0 / max(n0, 1)
            w1 = 1.0 / max(n1, 1)
            ws = [(w0 if (0 if self.all_pairs[i]["label"] == 2 else 1) == 0 else w1) for i in self.train_idx]
            sampler = WeightedRandomSampler(ws, num_samples=len(ws), replacement=True)
            train_loader = DataLoader(ds_train, batch_size=batch_size, sampler=sampler,
                                      pin_memory=True, num_workers=num_workers,
                                      persistent_workers=pw, drop_last=False)
        else:
            train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                                      pin_memory=True, num_workers=num_workers,
                                      persistent_workers=pw, drop_last=False)

        val_loader  = DataLoader(ds_val,  batch_size=val_bs, shuffle=False,
                                 pin_memory=True, num_workers=num_workers,
                                 persistent_workers=pw, drop_last=False)
        test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False,
                                 pin_memory=True, num_workers=num_workers,
                                 persistent_workers=pw, drop_last=False)
        return train_loader, val_loader, test_loader