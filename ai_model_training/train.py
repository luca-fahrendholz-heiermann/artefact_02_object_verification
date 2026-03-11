import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import os
import sys
import json
import multiprocessing
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Subset


dir_model = os.path.join(os.getcwd(), "model")
sys.path.append(dir_model)

from ov_ai_training import train_and_evaluate_scan2bim, train_and_evaluate, train_and_evaluate_siam, train_and_evaluate_cnn_xfeat, train_and_evaluate_cnn_5_channels, train_and_evaluate_cnn_xfeat_thresh, train_and_evaluate_cnn_xfeat_tr_87
from ov_ai_dataset import   ESFRefPairDatasetMLP704Fast, ESFRefPairDatasetChannels,ESFRefPairDatasetChannels_4_Xfeat, ESFRefPairDataset_siamese, ESFRefPairDatasetChannels_Xfeats, ESFRefPairDatasetChannels_5, ESFRefPairDatasetChannels_4, ESFRefPairDatasetChannels_4_Xfeat_S
from ov_ai_model import MLP704, MLP704v2,MLPFlex,  SiameseESFClassNet_NO_X,ESFResNetCNN_withVector_Att, SiameseESFClassNet, ESFResNetCNN, SiameseESFClassNet_Only_Xfeats, ESFResNetCNN_withVector
from ov_utils_mode import ChannelMaskWrapper, FeatMode
# Load Data
dir_data = os.path.join(os.getcwd(), "data")
dir_data_json = os.path.join(dir_data, "verf_esf_dataset_3_instances_merged.json")
dir_data_normal_json = os.path.join(dir_data, "verf_normal_xray_hist_dataset_3_instances_merged.json")
dir_data_extra_feats = os.path.join(dir_data, "features_memmap")  # <— ORDNER!
dir_data_extra_feats_grid = os.path.join(dir_data, "features_memmap_grid")  # <— ORDNER!
#dir_data_extra_feats = os.path.join(dir_data, "features_all_folds_aligned.json")
save_path_trained_model = os.path.join(os.getcwd(), "trained_model")
dir_cv_info_json = os.path.join(dir_data, 'cv6_info.json')
# "test_tanh_ep_3000_lr_0_001_4_layer_batch_norm"

def probe_batch(loader, d44=44, d27=27, device="cpu"):
    xb = next(iter(loader))
    x704, xext, y = xb  # deine Rückgabe
    if isinstance(xext, list):
        # falls Collate wegen variabler Länge eine Liste macht
        print("[WARN] xext ist Liste – mischt 44 und 71? Collate wird schwierig.")
        return
    print("[batch] x704", tuple(x704.shape), "xext", None if xext is None else tuple(xext.shape), "y", tuple(y.shape))
    if isinstance(xext, torch.Tensor):
        xext = xext.to(device)
        x44, x27 = _split_xext_debug(xext, d44, d27)
        if x44 is not None:
            print(f"extra44: mean={float(x44.mean()):.4f} std={float(x44.std()):.4f} nonzero%={(x44!=0).float().mean().item()*100:.2f}")
        if x27 is not None:
            print(f"grid27 : mean={float(x27.mean()):.4f} std={float(x27.std()):.4f} nonzero%={(x27!=0).float().mean().item()*100:.2f}")


def probe_samples(ds, k=5, d44=44, d27=27):
    from collections import Counter
    dims = Counter()
    for i in np.linspace(0, len(ds)-1, num=k, dtype=int):
        x704, xext, y = ds[i]
        print(f"[sample {i}] x704={tuple(x704.shape)} | xext={None if xext is None else tuple(xext.shape)} | y={int(y)}")
        if isinstance(xext, torch.Tensor):
            dims[xext.numel() if xext.ndim==1 else xext.size(-1)] += 1
            x44, x27 = _split_xext_debug(xext, d44, d27)
            if x44 is not None:
                print(f"  extra44: mean={float(x44.mean()):.4f} std={float(x44.std()):.4f} nonzero%={(x44!=0).float().mean().item()*100:.2f}")
            if x27 is not None:
                print(f"  grid27 : mean={float(x27.mean()):.4f} std={float(x27.std()):.4f} nonzero%={(x27!=0).float().mean().item()*100:.2f}")
    print("xext-dim counts:", dict(dims))

def _split_xext_debug(xext: torch.Tensor, d44=44, d27=27):
    """Zerlegt xext in (x44, x27) für Logging. Erwartet 44, 27 oder 71 Spalten."""
    if xext is None:
        return None, None
    if xext.ndim == 1:  # einzelnes Sample
        D = xext.size(0)
        if D >= d44 + d27:  # 71+
            return xext[:d44], xext[d44:d44+d27]
        if D == d44:
            return xext, None
        if D == d27:
            return None, xext
        return None, None
    # Batch
    B, D = xext.shape
    if D >= d44 + d27:
        return xext[:, :d44], xext[:, d44:d44+d27]
    if D == d44:
        return xext, None
    if D == d27:
        return None, xext
    return None, None


def main():
    # HYPERPARAMETER
    # Setup-Konfiguration definieren
    # 🚀 Setup-Konfiguration
    setup = {
        "learning_rate": 0.001,
        "dropout_rate": 0.5,
        "epochs": 500,
        "batch_size":64,
        "batch_size_val": 256,
        "output_size": 2,  # 2 für binär, 3 für 3-Klassen
        "pretraining": False,
        "num_worker": min(0, multiprocessing.cpu_count() // 2),
        "use_weighted_sampler": True
    }
    # 🔁 Cross-Validation
    cross_val = True

    print(setup["num_worker"], "/", multiprocessing.cpu_count() // 2)

    # Load Data
    dir_data = os.path.join(os.getcwd(), "data")
    precomputed_extra_feats_path = os.path.join(dir_data, "precomputed_features.jsonl")  # os.listdir(dir_data)[0])
    import orjson

    # Check if GPU is available
    print(torch.version.cuda)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Current device number is: {torch.cuda.current_device}")
    print(f"GPU name is {torch.cuda.get_device_name(torch.cuda.current_device)}")
    count = 0
    count_all = 0

    # Datenvorbereiten



    for fold_id in [5]:
        project = "obj_verf_2cl_cnn_grid_big_new_thr_extra"  # "test_tanh_ep_3000_lr_0_001_4_layer_batch_norm"
        #import gc

        # Garbage Collector explizit aufrufen
        #gc.collect()  # Dies erzwingt eine sofortige Speicherbereinigung

        if cross_val == True:
            #"""Load data from JSON file and extract cv info."""
            with open(dir_cv_info_json, 'r') as f:
                cv_info = json.load(f)
            fold = str(fold_id)

            project_name = f"{project}_fold_{fold}"


            # Create the dataset object with CV info
            print("Start Generating Dataset..")
            print(torch.cuda.memory_allocated() / 1024 ** 2, "MB allocated")
            print(torch.cuda.memory_reserved() / 1024 ** 2, "MB reserved")
            #dataset_with_cv = ESFRefPairDatasetChannels(dir_data_json, cv_info_path=dir_cv_info_json , fold="fold" + fold)
            #dataset_with_cv = ESFRefPairDataset_siamese(dir_data_json, cv_info_path=dir_cv_info_json , fold="fold" + fold, extra_feats_path=dir_data_extra_feats
            #dataset_with_cv = ESFRefPairDatasetChannels_Xfeats(dir_data_json, cv_info_path=dir_cv_info_json , fold="fold" + fold, extra_feats_path=dir_data_extra_feats)
            #dataset_with_cv = ESFRefPairDatasetChannels_4_Xfeat_S(dir_data_json, dir_data_normal_json,  cv_path=dir_cv_info_json ,
            #                                                    fold="fold" + fold,
            #                                                    extra_feats_path=dir_data_extra_feats)
            dataset_with_cv =  ESFRefPairDatasetMLP704Fast(dir_data_json, dir_data_normal_json,  cv_path=dir_cv_info_json ,
                                                               fold="fold" + fold, #drop_pos_perc_values={"30","35","40","45"},
                                                               drop_pos_where="scan", grid_feats_path = dir_data_extra_feats_grid,
                                                               extra_feats_path=dir_data_extra_feats, max_negatives_train=1000000)
            print("Start Generating Loader..")
            train_loader, val_loader, test_loader = dataset_with_cv.get_loaders(batch_size=setup["batch_size"], batch_size_val=setup["batch_size_val"], num_workers=setup["num_worker"])

            # --- Direkt testen ---
            print("\n[Probe TRAIN samples]")
            probe_samples(Subset(dataset_with_cv, dataset_with_cv.train_idx), k=5)
            print("\n[Probe VAL samples]")
            probe_samples(Subset(dataset_with_cv, dataset_with_cv.val_idx), k=5)

            print("\n[Probe first VAL batch]")
            probe_batch(val_loader, d44=44, d27=dataset_with_cv._grid_dim, device=device)

        else:
            # Case 2: Without Cross-Validation Info
            # Create the dataset object without CV info (just a train-test split)
            dataset_no_cv = ESFRefPairDatasetChannels_4_Xfeat_S(dir_data_json, cv_info_path=dir_cv_info_json, test_size=0.2, val_size=0.2)
            train_loader, val_loader, test_loader = dataset_no_cv.get_loaders(batch_size=setup["batch_size"], batch_size_val=setup["batch_size_val"], num_workers=setup["num_worker"], use_weighted_sampler=setup["use_weighted_sampler"])

        save_path = os.path.join(save_path_trained_model, project_name)
        print("End Generating Loader..")
        # Modell erzeugen

        # 2️⃣ Modell mit neuer Zielstruktur initialisieren (2 Klassen)
        # ➊ Mode wählen
        MODE = FeatMode.ALL  # "all" | "main" | "extra" | "grid" | main_extra | main_grid | extra_grid

        # ➋ Modell bauen (zum Mode passend; MLPFlex empfohlen)
        base_model = MLPFlex(p=setup["dropout_rate"], out_dim=setup["output_size"],
                             use_main=True,
                             use_esf_norm=True,
                             use_grid=True)

        #model = MLP704v2(p =setup["dropout_rate"], out_dim =setup["output_size"])
        #model = ESFResNetCNN_withVector_Att(num_classes=setup["output_size"], extra_dim=44)
        #model = SiameseESFClassNet_NO_X(num_classes=3, dropout_rate=setup["dropout_rate"])
        #model = SiameseESFClassNet(num_classes=3, dropout_rate=setup["dropout_rate"])
        #model = SiameseESFClassNet_Only_Xfeats(num_classes=3, dropout_rate=setup["dropout_rate"])
        #model = ESFResNetCNN_withVector(num_classes=2, dropout_rate=setup["dropout_rate"])

        # 5️⃣ Modell aufs Gerät verschieben
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # ➌ Wrapper drum
        model = ChannelMaskWrapper(base_model, mode=MODE, d_esf_norm=44, d_grid=27,d_main=704).to(device)

        # Modell trainieren und testen

        print(f'\n--- Training Setup: {setup} ---')
        if cross_val == True:
            #accuracy, precision, recall, f1 = train_and_evaluate_cnn_5_channels(model,dataset_with_cv,  train_loader, val_loader, test_loader, device = device, epochs=setup["epochs"], learning_rate=setup['learning_rate'], save_path=save_path, project= project,only_save_best_model = True, cv_info= cv_info, fold = "fold" + fold)
            #train_and_evaluate_cnn_xfeat_tr_87(model,dataset_with_cv,  train_loader, val_loader, test_loader, device = device, epochs=setup["epochs"], learning_rate=setup['learning_rate'], save_path=save_path, project= project,only_save_best_model = True, cv_info= cv_info, fold = "fold" + fold)
            accuracy, precision, recall, f1  = train_and_evaluate_cnn_xfeat_thresh(model,dataset_with_cv,  train_loader, val_loader, test_loader, device = device, epochs=setup["epochs"], learning_rate=setup['learning_rate'], save_path=save_path, project= project,only_save_best_model = True, cv_info= cv_info, fold = "fold" + fold)

            #accuracy, precision, recall, f1 = train_and_evaluate_scan2bim(model,  train_loader, val_loader, test_loader, device = device, epochs=setup["epochs"],learning_rate=setup['learning_rate'],save_path=save_path, project= project)
            #accuracy, precision, recall, f1 = train_and_evaluate(model, train_loader, val_loader, test_loader, device = device, epochs=setup["epochs"], learning_rate=setup['learning_rate'], save_path=save_path, project= project,only_save_best_model = True, cv_info= cv_info, fold = "fold" + fold)
        else:
            accuracy, precision, recall, f1 = train_and_evaluate(model, train_loader, val_loader, test_loader, device=device,
                                                                 epochs=setup["epochs"], learning_rate=setup['learning_rate'],
                                                                 save_path=save_path, project=project,
                                                                 only_save_best_model=True)
        # Ergebnisse des Setups ausdrucken
        print(f'\nResults for Setup: {setup}')
        print(f'- Test Accuracy: {accuracy*100:.2f}%')
        print(f'- Precision: {precision:.4f}')
        print(f'- Recall: {recall:.4f}')
        print(f'- F1-Score: {f1:.4f}')

if __name__ == "__main__":
    main()


