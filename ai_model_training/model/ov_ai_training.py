import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import os
import sys
import time
import json
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, precision_recall_fscore_support
)
import psutil, os
process = psutil.Process(os.getpid())
from ov_ai_model import FocalLoss, DualMetricScheduler, SoftF1Loss

from sklearn.metrics import precision_recall_fscore_support, f1_score


def combined_loss_function(outputs, labels, alpha=0.5):
    """
    Compute a combined loss using CrossEntropy and F1 loss.

    :param outputs: Model outputs (logits).
    :param labels: Ground truth labels.
    :param alpha: Weight for balancing the two losses (0.5 gives equal weight).
    :return: Combined loss.
    """
    # CrossEntropy Loss
    ce_loss = nn.CrossEntropyLoss()(outputs, labels)

    # F1 Loss (convert logits to probabilities)
    probs = torch.softmax(outputs, dim=1)
    preds = torch.argmax(probs, dim=1)

    tp = (preds == labels) & (labels != -1)  # True Positives
    fp = (preds != labels) & (labels != -1)  # False Positives
    fn = (preds != labels) & (labels != -1)  # False Negatives

    precision = tp.sum() / (tp.sum() + fp.sum() + 1e-8)
    recall = tp.sum() / (tp.sum() + fn.sum() + 1e-8)
    f1_loss = 1 - (2 * precision * recall) / (precision + recall + 1e-8)

    # Combine losses
    return alpha * ce_loss + (1 - alpha) * f1_loss

# Funktion zum Trainieren und Testen eines Modells
def train_and_evaluate_siam(model, train_loader, val_loader, test_loader, epochs=50, learning_rate=0.001, device = "cpu", save_path='./models', project = "default", only_save_best_model = False, cv_info=None, fold = None):
    # Move the model to the GPU (if available)
    model = model.to(device)

    #criterion = nn.CrossEntropyLoss(label_smoothing=0.1) # Output shape: [B, 3], Labels: LongTensor [B]
    class_weights = torch.tensor([1.0, 1.5, 0.5]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    #criterion = FocalLoss(alpha=[0.25, 0.75], gamma=2.0)
    #criterion = nn.BCEWithLogitsLoss() # binärer fall # Output muss shape [B, 1] sein → dann: labels = labels.float().unsqueeze(1)

    average_metric = "macro" # 'macro' , 'weighted

    # Focal Loss initialisieren
    #criterion = FocalLoss(alpha=0.25, gamma=2)

    #optimizer = optim.Adam(model.parameters(), lr=learning_rate)#,  weight_decay=0.0001)#, weight_decay=0.01) #optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9) #optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    #optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    scheduler = DualMetricScheduler(optimizer, patience=10, mode="max")

    #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=setup["learning_rate"], steps_per_epoch=len(train_loader),epochs=setup["epochs"], pct_start=0.1)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    # Ensure save directory exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # Definiere das Log-Verzeichnis
    logdir = os.path.join(save_path, 'logs')
    writer = tf.summary.create_file_writer(logdir)
    # tensorboard --logdir= logdir

    # Initialize lists to store history
    train_losses = []
    val_losses = []
    train_f1s = []
    val_f1s = []
    train_accuracies = []
    val_accuracies = []
    best_f1_score = 0
    best_model = None
    mini_batches = True
    accumulation_steps = 4  # Simuliert größere Batchgröße

    total_start_time = time.time()

    # Initialisiere die Subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    # Training Loop
    for epoch in range(epochs):
        epoch_start_time = time.time()  # Zeitmessung für die Epoche starten
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        all_labels_train = []
        all_predictions_train = []


        for i, (esf_ref, esf_scan, extra_features, labels) in enumerate(train_loader):
            # Move inputs and labels to GPU (if available)
            esf_ref = esf_ref.to(device)
            esf_scan = esf_scan.to(device)
            extra_features = extra_features.to(device)
            labels = labels.to(device)
            # Vorwärtsdurchlauf
            outputs = model(esf_ref, esf_scan, extra_features)
            loss = criterion(outputs, labels)
            #loss = combined_loss_function(outputs, labels, alpha=0.5)

            if mini_batches == True:
                loss = loss / accumulation_steps  # Loss normalisieren für Accumulation

            # Rückwärtsdurchlauf
            loss.backward()

            if (i + 1) % accumulation_steps == 0 and mini_batches == True:
                optimizer.step()
                optimizer.zero_grad() # Gradienten zurücksetzen

            # Ohne Gradient Accumulation: Standard-Optimierung
            if not mini_batches:
                optimizer.step()
                optimizer.zero_grad()
                running_loss += loss.item()

            else:
                # Laufenden Verlust berechnen
                running_loss += loss.item() * accumulation_steps  # Rückskalieren



            # Compute training accuracy
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            # Collect all labels and predictions for F1 score calculation
            all_labels_train.extend(labels.cpu().numpy())
            all_predictions_train.extend(predicted.cpu().numpy())

        train_accuracy = correct_train / total_train
        train_f1 = f1_score(all_labels_train, all_predictions_train, average=average_metric)

        train_losses.append(f"{epoch + 1}; {running_loss / len(train_loader):.4f}")
        train_accuracies.append(f"{epoch + 1}; {train_accuracy:.4f}")
        train_f1s.append(f"{epoch + 1}; {train_f1:.4f}")

        # Update train history files
        with open(os.path.join(save_path, f'{project}_train_loss.txt'), 'a') as f:
            f.write(f"{epoch + 1}; {running_loss / len(train_loader):.4f}\n")

        with open(os.path.join(save_path, f'{project}_train_accuracy.txt'), 'a') as f:
            f.write(f"{epoch + 1}; {train_accuracy:.4f}\n")

        with open(os.path.join(save_path, f'{project}_train_f1.txt'), 'a') as f:
            f.write(f"{epoch + 1}; {train_f1:.4f}\n")

        # Log metrics to TensorBoard
        with writer.as_default():
            tf.summary.scalar('Loss/train', running_loss / len(train_loader), step=epoch)
            tf.summary.scalar('Accuracy/train', train_accuracy, step=epoch)
            tf.summary.scalar('F1/train', train_f1, step=epoch)
            writer.flush()  # Sicherstellen, dass die Daten geschrieben werden



        # Validation Loop
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for (esf_ref, esf_scan, extra_features, labels) in val_loader:
                # Move inputs and labels to GPU (if available)
                esf_ref = esf_ref.to(device)
                esf_scan = esf_scan.to(device)
                extra_features = extra_features.to(device)
                labels = labels.to(device)

                outputs = model(esf_ref, esf_scan, extra_features)
                loss = criterion(outputs, labels)
                #loss = combined_loss_function(outputs, labels, alpha=0.5)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

                # Collect all labels and predictions for F1 score calculation
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

                # Scheduler Schritt nach der Epoche


        val_accuracy = correct_val / total_val

        #val_f1 = f1_score(all_labels, all_predictions, average='weighted')
        val_f1 = f1_score(all_labels, all_predictions, average=average_metric)
        # F1 scores for each label
        class_f1_scores = f1_score(all_labels, all_predictions, average=None)
        # Store F1 scores in a dictionary
        f1_scores_per_class = {f"Class {i}": f1 for i, f1 in enumerate(class_f1_scores)}

        # Example: Save to a file or log
        print("Class-wise F1 scores:", f1_scores_per_class)

        #scheduler.step(val_loss)
        scheduler.step(val_loss, val_f1)

        val_losses.append(f"{epoch + 1}; {val_loss / len(val_loader):.4f}")
        val_accuracies.append(f"{epoch + 1}; {val_accuracy:.4f}")
        val_f1s.append(f"{epoch + 1}; {val_f1:.4f}")



        # Update validation history files
        with open(os.path.join(save_path, f'{project}_val_loss.txt'), 'a') as f:
            f.write(f"{epoch + 1}; {val_loss / len(val_loader):.4f}\n")

        with open(os.path.join(save_path, f'{project}_val_accuracy.txt'), 'a') as f:
            f.write(f"{epoch + 1}; {val_accuracy:.4f}\n")

        with open(os.path.join(save_path, f'{project}_val_f1.txt'), 'a') as f:
            f.write(f"{epoch + 1}; {val_f1:.4f}\n")

        # Print metrics
        print(
            f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_losses[-1].split("; ")[1]}, Train Accuracy: {train_accuracy:.4f}, Train F1 Score: {train_f1:.4f}, Val Loss: {val_losses[-1].split("; ")[1]}, Val Accuracy: {val_accuracy:.4f}, Val F1 Score: {val_f1:.4f}')

        # Log validation metrics to TensorBoard
        with writer.as_default():
            tf.summary.scalar('Loss/val', val_loss / len(val_loader), step=epoch)
            tf.summary.scalar('Accuracy/val', val_accuracy, step=epoch)
            tf.summary.scalar('F1/val', val_f1, step=epoch)
            writer.flush()

        # Check if this is the best model based on F1 score
        if val_f1 > best_f1_score:
            best_f1_score = val_f1
            best_model = {
                'epoch': epoch + 1,
                'f1_score': best_f1_score,
                'model_state_dict': model.state_dict()
            }

            # Save the best model
            best_model_save_path = os.path.join(save_path, f'{project}_best_model.pth')
            torch.save(best_model["model_state_dict"], best_model_save_path)
            # print(f'Saved best model to {best_model_save_path}')

            print(f'New best model found with F1 Score: {best_f1_score:.4f} at epoch {epoch + 1}')

            # Save the best model info as JSON
            best_model_info = {
                'epoch': best_model['epoch'],
                'f1_score': best_model['f1_score'],
                'model_save_path': best_model_save_path
            }
            best_model_json_path = os.path.join(save_path, f'{project}_best_model_info.json')
            with open(best_model_json_path, 'w') as json_file:
                json.dump(best_model_info, json_file)

        # Save the model after each epoch
        if only_save_best_model == False:
            model_save_path = os.path.join(save_path, f'{project}_model_epoch_{epoch + 1}_acc_{np.round(val_accuracy,4)}_f1_{np.round(val_f1,4)}.pth')
            torch.save(model.state_dict(), model_save_path)
        #print(f'Saved model to {model_save_path}')




        #print(f'Saved best model info to {best_model_json_path}')

        # Zeitmessung für die Epoche stoppen
        epoch_time = time.time() - epoch_start_time
        #print(f"Time taken for epoch {epoch + 1}: {epoch_time:.2f} seconds")

        # Speichern der Zeit für die Epoche in einer Datei
        with open(os.path.join(save_path, f'{project}_epoch_times.txt'), 'a') as f:
            f.write(f"Epoch {epoch + 1}: {epoch_time:.2f} seconds\n")

        #plotten







        # Gesamtzeit des Trainings messen
    total_training_time = time.time() - total_start_time
    print(f"Total training time: {total_training_time:.2f} seconds")

    # Speichern der Gesamtzeit in einer Datei
    with open(os.path.join(save_path, f'{project}_total_training_time.txt'), 'w') as f:
        f.write(f"Total training time: {total_training_time:.2f} seconds\n")

    print("Start Test Loop")
    # Test Loop
    # --- Test ---
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for batch_idx, (esf_diff, extra_features, labels) in enumerate(test_loader):
            esf_diff = esf_diff.to(device)
            extra_features = extra_features.to(device)
            labels = labels.to(device)

            outputs = model(esf_diff, extra_features)
            _, predicted = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(predicted.cpu().numpy().tolist())

    # numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Grundmetriken (macro, wie bei dir)
    test_accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # Confusion-Matrix + per-class P/R/F1
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TP = np.diag(cm)
    FP = cm.sum(axis=0) - TP
    FN = cm.sum(axis=1) - TP
    TN = cm.sum() - (TP + FP + FN)

    # per-class via sklearn (bequem & robust)
    prec, rec, f1_cls, supp = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )

    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Precision (macro): {precision_macro:.4f}")
    print(f"Recall (macro):    {recall_macro:.4f}")
    print(f"F1-Score (macro):  {f1_macro:.4f}")
    print("Confusion Matrix:")
    print(cm)

    print("Class-wise metrics (TEST):")
    print(f"  Class 0 -> P={prec[0]:.4f} | R={rec[0]:.4f} | F1={f1_cls[0]:.4f} | n={supp[0]}")
    print(f"  Class 1 -> P={prec[1]:.4f} | R={rec[1]:.4f} | F1={f1_cls[1]:.4f} | n={supp[1]}")

    # (optional) Kritische Fehler explizit ausgeben: 0->1 = FP für Klasse 1
    fp_class1 = cm[0, 1]
    print(f"Kritische Fehler (0→1): {fp_class1}")

    # ---- Excel speichern (NEUER Name für Writer! Kein flush hier!) ----
    conf_matrix_df = pd.DataFrame(
        cm,
        index=[f"True_{i}" for i in [0, 1]],
        columns=[f"Pred_{i}" for i in [0, 1]]
    )

    metrics = {
        "Class": [0, 1],
        "Support": supp,
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
        "Precision": prec,
        "Recall": rec,
        "F1": f1_cls,
    }
    df_metrics = pd.DataFrame(metrics)

    summary = pd.DataFrame({
        "Metric": ["Accuracy", "Precision(macro)", "Recall(macro)", "F1(macro)", "FP_class1(0→1)"],
        "Value": [test_accuracy, precision_macro, recall_macro, f1_macro, fp_class1]
    })

    excel_path = os.path.join(save_path, f'{project}_test_metrics.xlsx')
    with pd.ExcelWriter(excel_path, engine='xlsxwriter') as xls_writer:
        df_metrics.to_excel(xls_writer, sheet_name='Per_Class', index=False)
        conf_matrix_df.to_excel(xls_writer, sheet_name='Confusion_Matrix')
        summary.to_excel(xls_writer, sheet_name='Summary', index=False)

    print(f"Saved test metrics to {excel_path}")


from torch.utils.data import SubsetRandomSampler, DataLoader, Subset

def _y_mapped_from_pair(p):
    # 0 => verwerfen; 2 => 0; 1 => 1
    y = p["label"]
    if y == 0:
        return None
    return 0 if y == 2 else 1

from torch.utils.data import WeightedRandomSampler

def make_epoch_sampler(
    train_idx,
    all_pairs,
    max_negatives=500_000,
    keep_all_positives=True,
    hard_neg=None,         # <- neu: Set von Indexen
    boost=5.0              # <- neu: Gewicht für harte 0er
):
    hard_neg = hard_neg or set()

    pos_idx = []
    neg_idx = []
    for i in train_idx:
        ym = _y_mapped_from_pair(all_pairs[i])
        if ym is None:
            continue
        if ym == 1:
            pos_idx.append(i)
        else:
            neg_idx.append(i)

    # Negatives subsamplen (vor Gewichtung)
    if max_negatives is not None and len(neg_idx) > max_negatives:
        neg_sampled = np.random.choice(neg_idx, size=max_negatives, replace=False)
    else:
        neg_sampled = np.asarray(neg_idx, dtype=np.int64)

    # Positives behalten
    if keep_all_positives:
        indices = np.concatenate([np.asarray(pos_idx, dtype=np.int64), neg_sampled])
    else:
        indices = neg_sampled

    # Gewichte bauen (harte 0er bekommen Boost)
    weights = np.ones(len(indices), dtype=np.float32)
    for k, idx in enumerate(indices):
        if idx in hard_neg:
            weights[k] = boost
        # optional: Positives leicht boosten, wenn du willst
        # elif idx in pos_idx_set: weights[k] = 1.5

    # Sampler mit Replacement (damit Gewicht wirkt)
    sampler = WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.float32),
        num_samples=len(indices),
        replacement=True
    )
    return sampler, indices  # indices brauchst du für Subset


def compute_class_weights_from_loader(train_loader, num_classes=2, alpha0=1.0, device="cpu"):
    import torch
    counts = torch.zeros(num_classes, dtype=torch.long)
    for *_, y in train_loader:
        y = y.view(-1).to("cpu")
        counts += torch.bincount(y, minlength=num_classes)

    N = counts.sum().item()
    base = (N / (num_classes * counts.float().clamp_min(1))).to(torch.float32)

    # Kosten-Prior: Klasse 0 teurer -> weniger FP(1)
    cost = torch.tensor([alpha0, 1.0], dtype=torch.float32)
    w = base * cost

    # auf Mittelwert 1 normieren (optional, verhindert zu große Gradienten)
    w = w * (num_classes / w.sum())
    print(f"[weights] counts={counts.tolist()} -> weights={w.tolist()}")
    return w.to(device)

def count_classes_from_indices(dataset, indices, num_classes=2):
    import torch
    cnt = torch.zeros(num_classes, dtype=torch.long)
    for i in indices:
        y = dataset._y_mapped(dataset.all_pairs[i])
        if y is None:
            continue
        cnt[y] += 1
    return cnt

def make_ce_weights_from_dataset(dataset, alpha_neg=1.3, device="cpu"):
    counts = count_classes_from_indices(dataset, dataset.train_idx, 2)  # [n0, n1]
    N = counts.sum().item()
    base = (N / (2 * counts.float().clamp_min(1)))   # inverse freq
    cost = torch.tensor([alpha_neg, 1.0])            # 0 teurer
    w = base * cost
    w = w * (2.0 / w.sum())                          # Mittel=1
    print(f"[weights] raw_counts={counts.tolist()} -> ce_weights={w.tolist()}")
    return w.to(device)  # <<< NICHT mehr drehen


from torch.utils.data import Subset
def train_and_evaluate_cnn_xfeat_back(model, dataset_with_cv, train_loader, val_loader, test_loader, epochs=50, learning_rate=0.001, device = "cpu", save_path='./models', project = "default", only_save_best_model = False, cv_info=None, fold = None):
    # Move the model to the GPU (if available)
    model = model.to(device)

    #criterion = nn.CrossEntropyLoss(label_smoothing=0.1) # Output shape: [B, 3], Labels: LongTensor [B]
    #class_weights = torch.tensor([1.0, 1.5, 0.5]).to(device)
    class_weights = make_ce_weights_from_dataset(dataset_with_cv, alpha_neg=1.6, device=device)

    criterion = nn.CrossEntropyLoss( label_smoothing=0.0)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.0)
    #criterion = FocalLoss(alpha=[0.25, 0.75], gamma=2.0)
    #criterion = nn.BCEWithLogitsLoss() # binärer fall # Output muss shape [B, 1] sein → dann: labels = labels.float().unsqueeze(1)

    average_metric = "macro" # 'macro' , 'weighted

    # Focal Loss initialisieren
    #criterion = FocalLoss(alpha=0.25, gamma=2)

    #optimizer = optim.Adam(model.parameters(), lr=learning_rate)#,  weight_decay=0.0001)#, weight_decay=0.01) #optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9) #optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    #optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    scheduler = DualMetricScheduler(optimizer, patience=10, mode="max")

    #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=setup["learning_rate"], steps_per_epoch=len(train_loader),epochs=setup["epochs"], pct_start=0.1)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    # Ensure save directory exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # Definiere das Log-Verzeichnis
    logdir = os.path.join(save_path, 'logs')
    writer = tf.summary.create_file_writer(logdir)
    # tensorboard --logdir= logdir

    # Initialize lists to store history
    train_losses = []
    val_losses = []
    train_f1s = []
    val_f1s = []
    train_accuracies = []
    val_accuracies = []
    best_f1_score = 0
    best_model = None
    mini_batches = True
    accumulation_steps = 4  # Simuliert größere Batchgröße

    total_start_time = time.time()

    # Initialisiere die Subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    # Training Loop
    for epoch in range(epochs):
        sampler = make_epoch_sampler(dataset_with_cv.train_idx,
                                     dataset_with_cv.all_pairs,
                                     max_negatives=1000000)

        train_loader = DataLoader(Subset(dataset_with_cv, dataset_with_cv.train_idx),
                                  batch_size=128,
                                  sampler=sampler,
                                  num_workers=0,
                                  drop_last=False)
        epoch_start_time = time.time()  # Zeitmessung für die Epoche starten
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        all_labels_train = []
        all_predictions_train = []


        for i, (esf_diff, extra_features, labels) in enumerate(train_loader):
            # Move inputs and labels to GPU (if available)
            esf_diff = esf_diff.to(device)
            extra_features = extra_features.to(device)
            labels = labels.to(device)
            # Vorwärtsdurchlauf
            outputs = model(esf_diff, extra_features)
            loss = criterion(outputs, labels)
            #loss = combined_loss_function(outputs, labels, alpha=0.5)

            if mini_batches == True:
                loss = loss / accumulation_steps  # Loss normalisieren für Accumulation

            # Rückwärtsdurchlauf
            loss.backward()

            if (i + 1) % accumulation_steps == 0 and mini_batches == True:
                optimizer.step()
                optimizer.zero_grad() # Gradienten zurücksetzen

            # Ohne Gradient Accumulation: Standard-Optimierung
            if not mini_batches:
                optimizer.step()
                optimizer.zero_grad()
                running_loss += loss.item()

            else:
                # Laufenden Verlust berechnen
                running_loss += loss.item() * accumulation_steps  # Rückskalieren



            # Compute training accuracy
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            # Collect all labels and predictions for F1 score calculation
            all_labels_train.extend(labels.cpu().numpy())
            all_predictions_train.extend(predicted.cpu().numpy())

        train_accuracy = correct_train / total_train
        train_f1 = f1_score(all_labels_train, all_predictions_train, average=average_metric)

        train_losses.append(f"{epoch + 1}; {running_loss / len(train_loader):.4f}")
        train_accuracies.append(f"{epoch + 1}; {train_accuracy:.4f}")
        train_f1s.append(f"{epoch + 1}; {train_f1:.4f}")

        # Update train history files
        with open(os.path.join(save_path, f'{project}_train_loss.txt'), 'a') as f:
            f.write(f"{epoch + 1}; {running_loss / len(train_loader):.4f}\n")

        with open(os.path.join(save_path, f'{project}_train_accuracy.txt'), 'a') as f:
            f.write(f"{epoch + 1}; {train_accuracy:.4f}\n")

        with open(os.path.join(save_path, f'{project}_train_f1.txt'), 'a') as f:
            f.write(f"{epoch + 1}; {train_f1:.4f}\n")

        # Log metrics to TensorBoard
        with writer.as_default():
            tf.summary.scalar('Loss/train', running_loss / len(train_loader), step=epoch)
            tf.summary.scalar('Accuracy/train', train_accuracy, step=epoch)
            tf.summary.scalar('F1/train', train_f1, step=epoch)
            writer.flush()  # Sicherstellen, dass die Daten geschrieben werden

        # Validation Loop
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for (esf_diff, extra_features, labels) in val_loader:
                esf_diff = esf_diff.to(device)
                extra_features = extra_features.to(device)
                labels = labels.to(device)

                outputs = model(esf_diff, extra_features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        val_accuracy = correct_val / total_val

        # Gesamt-F1 (macro/weighted je nach deiner Variable)
        val_f1 = f1_score(all_labels, all_predictions, average=average_metric)

        # Per-Klasse: Precision/Recall/F1 + Support
        prec, rec, f1_cls, supp = precision_recall_fscore_support(
            all_labels, all_predictions, labels=[0, 1], zero_division=0
        )

        print("Class-wise metrics (VAL):")
        print(f"  Class 0 -> P={prec[0]:.4f} | R={rec[0]:.4f} | F1={f1_cls[0]:.4f} | n={supp[0]}")
        print(f"  Class 1 -> P={prec[1]:.4f} | R={rec[1]:.4f} | F1={f1_cls[1]:.4f} | n={supp[1]}")

        # Falls du das alte Dict behalten willst:
        class_f1_scores = f1_cls
        f1_scores_per_class = {f"Class {i}": float(s) for i, s in enumerate(class_f1_scores)}
        print("Class-wise F1 scores:", f1_scores_per_class)

        # Scheduler & Logging wie gehabt
        scheduler.step(val_loss, val_f1)

        val_losses.append(f"{epoch + 1}; {val_loss / len(val_loader):.4f}")
        val_accuracies.append(f"{epoch + 1}; {val_accuracy:.4f}")
        val_f1s.append(f"{epoch + 1}; {val_f1:.4f}")

        with open(os.path.join(save_path, f'{project}_val_loss.txt'), 'a') as f:
            f.write(f"{epoch + 1}; {val_loss / len(val_loader):.4f}\n")
        with open(os.path.join(save_path, f'{project}_val_accuracy.txt'), 'a') as f:
            f.write(f"{epoch + 1}; {val_accuracy:.4f}\n")
        with open(os.path.join(save_path, f'{project}_val_f1.txt'), 'a') as f:
            f.write(f"{epoch + 1}; {val_f1:.4f}\n")

        print(
            f"Epoch {epoch + 1}/{epochs}, "
            f"Train Loss: {train_losses[-1].split('; ')[1]}, "
            f"Train Accuracy: {train_accuracy:.4f}, Train F1 Score: {train_f1:.4f}, "
            f"Val Loss: {val_losses[-1].split('; ')[1]}, Val Accuracy: {val_accuracy:.4f}, Val F1 Score: {val_f1:.4f}"
        )

        with writer.as_default():
            tf.summary.scalar('Loss/val', val_loss / len(val_loader), step=epoch)
            tf.summary.scalar('Accuracy/val', val_accuracy, step=epoch)
            tf.summary.scalar('F1/val', val_f1, step=epoch)
            # Optional: per Klasse loggen
            tf.summary.scalar('Val/Class0_Precision', prec[0], step=epoch)
            tf.summary.scalar('Val/Class0_Recall', rec[0], step=epoch)
            tf.summary.scalar('Val/Class1_Precision', prec[1], step=epoch)
            tf.summary.scalar('Val/Class1_Recall', rec[1], step=epoch)
            writer.flush()

        # Check if this is the best model based on F1 score
        if val_f1 > best_f1_score:
            best_f1_score = val_f1
            best_model = {
                'epoch': epoch + 1,
                'f1_score': best_f1_score,
                'model_state_dict': model.state_dict()
            }

            # Save the best model
            best_model_save_path = os.path.join(save_path, f'{project}_best_model.pth')
            torch.save(best_model["model_state_dict"], best_model_save_path)
            # print(f'Saved best model to {best_model_save_path}')

            print(f'New best model found with F1 Score: {best_f1_score:.4f} at epoch {epoch + 1}')

            # Save the best model info as JSON
            best_model_info = {
                'epoch': best_model['epoch'],
                'f1_score': best_model['f1_score'],
                'model_save_path': best_model_save_path
            }
            best_model_json_path = os.path.join(save_path, f'{project}_best_model_info.json')
            with open(best_model_json_path, 'w') as json_file:
                json.dump(best_model_info, json_file)

        # Save the model after each epoch
        if only_save_best_model == False:
            model_save_path = os.path.join(save_path, f'{project}_model_epoch_{epoch + 1}_acc_{np.round(val_accuracy,4)}_f1_{np.round(val_f1,4)}.pth')
            torch.save(model.state_dict(), model_save_path)
        #print(f'Saved model to {model_save_path}')




        #print(f'Saved best model info to {best_model_json_path}')

        # Zeitmessung für die Epoche stoppen
        epoch_time = time.time() - epoch_start_time
        #print(f"Time taken for epoch {epoch + 1}: {epoch_time:.2f} seconds")

        # Speichern der Zeit für die Epoche in einer Datei
        with open(os.path.join(save_path, f'{project}_epoch_times.txt'), 'a') as f:
            f.write(f"Epoch {epoch + 1}: {epoch_time:.2f} seconds\n")

        #plotten







        # Gesamtzeit des Trainings messen
    total_training_time = time.time() - total_start_time
    print(f"Total training time: {total_training_time:.2f} seconds")

    # Speichern der Gesamtzeit in einer Datei
    with open(os.path.join(save_path, f'{project}_total_training_time.txt'), 'w') as f:
        f.write(f"Total training time: {total_training_time:.2f} seconds\n")

    print("Start Test Loop")
    # Test Loop
    model.eval()
    y_true = []
    y_pred = []
    # Initialisiere eine Liste, um die globalen Indizes zu speichern
    global_test_indices = []
    wrong_indices_by_fold = {}
    with torch.no_grad():
        for batch_idx, (esf_diff, extra_features, labels) in enumerate(test_loader):
            # Move inputs and labels to GPU (if available)
            esf_diff = esf_diff.to(device)
            extra_features = extra_features.to(device)
            labels = labels.to(device)

            outputs = model(esf_diff, extra_features)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

            if cv_info != None:
                wrong_predictions = predicted != labels
                # Extrahiere die Batch-individuellen Indizes
                batch_wrong_indices = torch.nonzero(wrong_predictions).squeeze()

                if batch_wrong_indices.numel() == 0:  # Prüfen, ob die Liste leer ist
                    print(f"Alle Vorhersagen im Batch {batch_idx} sind korrekt.")
                    continue  # Überspringe diesen Batch

                # Berechne die globalen Indizes innerhalb des Testloaders
                global_indices = batch_idx * test_loader.batch_size + batch_wrong_indices.cpu().numpy()

                if isinstance(global_indices, np.int64):
                    global_test_indices.extend([global_indices])
                    #print(batch_idx, global_test_indices)
                else:
                    global_test_indices.extend(global_indices)
                    #print(batch_idx, global_test_indices)


        global_test_indices  =  list(global_test_indices)




        print("wrong data labeled: ", len(global_test_indices))

        # Metriken berechnen
        test_accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average=average_metric)
        recall = recall_score(y_true, y_pred, average=average_metric)
        f1 = f1_score(y_true, y_pred, average=average_metric)

        # F1 scores for each label
        class_f1_scores = f1_score(y_true, y_pred, average=None)
        # Store F1 scores in a dictionary
        f1_scores_per_class = {f"Class {i}": f1 for i, f1 in enumerate(class_f1_scores)}

        # Example: Save to a file or log
        print("Class-wise F1 scores:", f1_scores_per_class)

        if cv_info != None:
            # Extrahiere die ursprünglichen Test-Indizes aus cv_info
            original_test_indices = cv_info[fold]["test"]

            # Zuordnung zu ursprünglichen Indizes im Datensatz
            original_wrong_indices = [original_test_indices[idx] for idx in global_test_indices]
            wrong_indices_by_fold[fold] = original_wrong_indices
            file_path = "wrong_labeled_data.json"  # Specify the file name or path
            with open(file_path, "a") as json_file:
                json.dump(wrong_indices_by_fold, json_file, indent=4)  # indent=4 for pretty formatting

        # Confusion Matrix
        conf_matrix = confusion_matrix(y_true, y_pred)
        TP = np.diag(conf_matrix)  # True Positives für jede Klasse
        FP = conf_matrix.sum(axis=0) - TP  # False Positives für jede Klasse
        FN = conf_matrix.sum(axis=1) - TP  # False Negatives für jede Klasse
        TN = conf_matrix.sum() - (FP + FN + TP)  # True Negatives für jede Klasse

        # Convert the confusion matrix to a DataFrame
        conf_matrix_df = pd.DataFrame(
            conf_matrix,
            index=[f"True_{i}" for i in range(conf_matrix.shape[0])],  # Rows: True labels
            columns=[f"Pred_{i}" for i in range(conf_matrix.shape[1])]  # Columns: Predicted labels
        )

        # Ergebnisse ausgeben
        print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1-Score: {f1:.4f}')
        print('Confusion Matrix:')
        print(conf_matrix)

        print(f'True Positives (TP): {TP}')
        print(f'False Positives (FP): {FP}')
        print(f'False Negatives (FN): {FN}')
        print(f'True Negatives (TN): {TN}')

        # Metriken berechnen für jede Klasse
        precision_per_class = TP / (TP + FP)
        recall_per_class = TP / (TP + FN)
        f1_per_class = 2 * (precision_per_class * recall_per_class) / (precision_per_class + recall_per_class)
        accuracy_per_class = (TP + TN) / (TP + TN + FP + FN)

        # Gesamtmetriken berechnen (gewichtete Durchschnittswerte über alle Klassen)
        total_TP = TP.sum()
        total_FP = FP.sum()
        total_FN = FN.sum()
        total_TN = TN.sum()

        total_precision = total_TP / (total_TP + total_FP)
        total_recall = total_TP / (total_TP + total_FN)
        total_f1 = 2 * (total_precision * total_recall) / (total_precision + total_recall)
        total_accuracy = (total_TP + total_TN) / (total_TP + total_TN + total_FP + total_FN)

        # Ergebnisse in DataFrame speichern
        metrics = {
            'Class': np.arange(len(TP)),
            'True Positives': TP,
            'False Positives': FP,
            'False Negatives': FN,
            'True Negatives': TN,
            'Precision': precision_per_class,
            'Recall': recall_per_class,
            'F1-Score': f1_per_class,
            'Accuracy': accuracy_per_class
        }

        df_metrics = pd.DataFrame(metrics)

        # Gesamtmetriken hinzufügen
        total_metrics = {
            'Class': 'Total',
            'True Positives': total_TP,
            'False Positives': total_FP,
            'False Negatives': total_FN,
            'True Negatives': total_TN,
            'Precision': total_precision,
            'Recall': total_recall,
            'F1-Score': total_f1,
            'Accuracy': total_accuracy
        }

        df_total_metrics = pd.DataFrame(total_metrics, index=[0])
        df_metrics = pd.concat([df_metrics, df_total_metrics], ignore_index=True)

        # DataFrame in eine Excel-Datei speichern
        excel_path = os.path.join(save_path, f'{project}_test_metrics.xlsx')
        with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
            df_metrics.to_excel(writer, sheet_name='Metrics_Per_Class', index=False)
            conf_matrix_df.to_excel(writer, sheet_name='Confusion_Matrix')

        print(f'Saved test metrics to {excel_path}')


    return test_accuracy, precision, recall, f1
#######
def train_and_evaluate_cnn_xfeat_tr_87(
    model,
    dataset_with_cv,
    train_loader,      # initialer Loader (nur für steps_per_epoch von OneCycle)
    val_loader,
    test_loader,
    epochs=50,
    learning_rate=3e-4,
    device="cpu",
    save_path="./models",
    project="default",
    only_save_best_model=True,
    cv_info=None,
    fold=None,
    # ---- Threshold-Optionen
    fp0_max_rate=0.006,             # max erlaubte 0->1-Fehlrate (FPR0) auf VAL
    thresh_grid=None,               # z.B. np.linspace(0.5, 0.99, 100)
    thresh_objective="macro_f1",    # "macro_f1" | "f1_class1" | "recall_class1"
    # ---- Hard-Negative-Mining
    max_negatives_per_epoch=500_000,
    boost_hard_neg=5.0,
    topk_hard_neg=50_000,
    # ---- Class-Weights
    class_weights_vec=None,         # z.B. [4.0, 1.0]; None => auto versuchen
    weight_decay=1e-5,
    grad_accum_steps=4,
):
    """
    Trainiert mit Hard-Negatives und wählt pro Epoche einen Threshold t* auf der Validation,
    der die Bedingung FPR0 <= fp0_max_rate erfüllt und nach thresh_objective optimiert.
    Speichert das Best-Model (nach F1@t*) und exportiert Testmetriken inkl. per-Klasse.
    """
    import os, time, json, gc
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
    from sklearn.metrics import (
        confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
        precision_recall_fscore_support
    )

    # optional: RAM-Monitor
    try:
        import psutil
        process = psutil.Process(os.getpid())
    except Exception:
        process = None

    # optional: TensorBoard (nur CPU-Benutzung, kein GPU-Claim)
    try:
        import tensorflow as tf
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass
    except Exception:
        tf = None

    os.makedirs(save_path, exist_ok=True)
    writer = tf.summary.create_file_writer(os.path.join(save_path, "logs")) if tf is not None else None

    # ---------- Hilfsfunktionen ----------
    def _get_y_mapped_from_pair(p):
        # nimmt dein Mapping, falls vorhanden; sonst Fallback
        try:
            return globals()["_y_mapped_from_pair"](p)
        except Exception:
            y = p.get("label", None)
            if y is None or y == 0:
                return None
            return 0 if y == 2 else 1

    def _label_from_pair_idx(i):
        ym = _get_y_mapped_from_pair(dataset_with_cv.all_pairs[i])
        return None if ym is None else int(ym)

    def metrics_from_counts(tn, fp, fn, tp):
        # per Klasse und macro
        prec1 = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec1  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f11   = (2*prec1*rec1)/(prec1+rec1) if (prec1+rec1) > 0 else 0.0

        prec0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        rec0  = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f10   = (2*prec0*rec0)/(prec0+rec0) if (prec0+rec0) > 0 else 0.0

        macro_f1 = 0.5 * (f10 + f11)
        return (prec0, rec0, f10), (prec1, rec1, f11), macro_f1

    def pick_threshold(p1, y, grid, max_fp0_rate, objective="macro_f1"):
        # gibt best_t, dict(metrics) zurück
        if grid is None or len(grid) == 0:
            grid = np.linspace(0.5, 0.99, 100)

        best = None
        best_key = None

        # pre-counts für 0/1
        for t in grid:
            pred = (p1 >= t).astype(np.int32)
            cm = confusion_matrix(y, pred, labels=[0, 1])
            if cm.shape != (2, 2):
                tn = fp = fn = tp = 0
                if cm.shape == (1, 1):  # nur eine Klasse gesehen
                    tn = int(cm[0, 0]) if (y == 0).all() else 0
                    tp = int(cm[0, 0]) if (y == 1).all() else 0
                # ansonsten leeres/degeneriertes Setting
            else:
                tn, fp, fn, tp = int(cm[0,0]), int(cm[0,1]), int(cm[1,0]), int(cm[1,1])

            # FPR0 = 0->1-Rate
            denom = tn + fp
            fpr0 = (fp / denom) if denom > 0 else 0.0

            (p0,r0,f10),(p1c,r1c,f11),macro_f1 = metrics_from_counts(tn, fp, fn, tp)

            # Auswahl-Key
            if objective == "f1_class1":
                key_primary = f11
            elif objective == "recall_class1":
                key_primary = r1c
            else:
                key_primary = macro_f1

            # Nebenbedingung
            feasible = (fpr0 <= max_fp0_rate)

            # Sortierschlüssel: erst Feasibility, dann primary, dann (–fpr0), dann (–t) konservativ hoch
            key = (1 if feasible else 0, key_primary, -fpr0, -t)

            cand = {
                "t": float(t),
                "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                "fpr0": float(fpr0),
                "macro_f1": float(macro_f1),
                "p0": float(p0), "r0": float(r0), "f10": float(f10),
                "p1": float(p1c), "r1": float(r1c), "f11": float(f11),
                "feasible": bool(feasible)
            }
            if (best_key is None) or (key > best_key):
                best = cand
                best_key = key

        return best["t"], best

    def make_epoch_sampler(train_idx, all_pairs, max_negatives, keep_all_positives=True, hard_neg=None, boost=5.0):
        hard_neg = hard_neg or set()
        pos_idx, neg_idx = [], []
        for i in train_idx:
            ym = _get_y_mapped_from_pair(all_pairs[i])
            if ym is None:  # gelabelte 0 (alt) wird verworfen
                continue
            (pos_idx if ym == 1 else neg_idx).append(i)

        if (max_negatives is not None) and (len(neg_idx) > max_negatives):
            neg_sampled = np.random.choice(neg_idx, size=max_negatives, replace=False)
        else:
            neg_sampled = np.asarray(neg_idx, dtype=np.int64)

        indices = np.asarray(pos_idx, dtype=np.int64) if keep_all_positives else np.array([], dtype=np.int64)
        if len(neg_sampled) > 0:
            indices = np.concatenate([indices, neg_sampled])

        # Gewichte (Hard-Negs boosten)
        weights = np.ones(len(indices), dtype=np.float32)
        hard_set = set(hard_neg)
        for k, idx in enumerate(indices):
            if idx in hard_set:
                weights[k] = float(boost)

        sampler = WeightedRandomSampler(
            weights=torch.tensor(weights, dtype=torch.float32),
            num_samples=len(indices),
            replacement=True
        )
        return sampler, indices

    # ---------- Setup ----------
    device = torch.device(device)
    model = model.to(device)

    # Class-Weights
    if class_weights_vec is not None:
        class_weights = torch.tensor(class_weights_vec, dtype=torch.float32, device=device)
    else:
        # auto: versuche helper aus deiner Umgebung
        try:
            class_weights = make_ce_weights_from_dataset(dataset_with_cv, alpha_neg=1.6, device=device)
        except Exception:
            # Fallback neutral
            class_weights = torch.tensor([1.0, 1.0], dtype=torch.float32, device=device)
    print("[CE weights]", class_weights.tolist())

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # OneCycle – steps_per_epoch von initialem train_loader
    steps_per_epoch = max(1, len(train_loader))
    onecycle = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=max(learning_rate, 1e-3), epochs=epochs,
        steps_per_epoch=steps_per_epoch, div_factor=25.0, final_div_factor=1e4
    )
    # Später auf F1@t* umschalten
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, threshold=1e-4,
        cooldown=1, min_lr=1e-6
    )
    use_plateau_after = max(4, epochs - 4)

    # History
    best_f1 = -1.0
    best_info = None
    best_state = None

    hard_neg_prev = set()
    base_bs = getattr(train_loader, "batch_size", 64)

    # ---------- Training ----------
    total_start = time.time()

    for epoch in range(epochs):
        # (1) Epoche: Sampler bauen
        try:
            sampler, ep_indices = make_epoch_sampler(
                dataset_with_cv.train_idx,
                dataset_with_cv.all_pairs,
                max_negatives=max_negatives_per_epoch,
                keep_all_positives=True,
                hard_neg=hard_neg_prev,
                boost=boost_hard_neg
            )
            tr_loader = DataLoader(
                Subset(dataset_with_cv, ep_indices),
                batch_size=base_bs, sampler=sampler, num_workers=0, drop_last=False
            )
        except Exception as e:
            print(f"[Sampler-Fallback] {e}")
            tr_loader = train_loader  # initialer

        # (2) Train
        model.train()
        running_loss = 0.0
        tn_tr = fp_tr = fn_tr = tp_tr = 0
        optimizer.zero_grad(set_to_none=True)
        accum = 0
        num_batches = len(tr_loader)

        for b, (x704, xext, labels) in enumerate(tr_loader, start=1):
            x704   = x704.to(device, non_blocking=True)
            xext   = xext.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(x704, xext)
            loss = criterion(logits, labels)

            loss = loss / max(1, grad_accum_steps)
            loss.backward()
            accum += 1

            # Optimizer-Schritt
            if (accum == grad_accum_steps) or (b == num_batches):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                onecycle.step()
                accum = 0

            running_loss += loss.item() * max(1, grad_accum_steps)

            # Online-Counts
            with torch.no_grad():
                pred = logits.argmax(1)
                cm = confusion_matrix(labels.detach().cpu().numpy(),
                                      pred.detach().cpu().numpy(), labels=[0, 1])
                if cm.shape == (2,2):
                    tn_tr += int(cm[0,0]); fp_tr += int(cm[0,1])
                    fn_tr += int(cm[1,0]); tp_tr += int(cm[1,1])

            del logits, loss, pred
            if torch.cuda.is_available() and (b % 100 == 0):
                torch.cuda.empty_cache()

        train_total = tn_tr + fp_tr + fn_tr + tp_tr
        train_acc = (tn_tr + tp_tr) / max(1, train_total)
        (p0_t, r0_t, f10_t), (p1_t, r1_t, f11_t), macro_f1_t = metrics_from_counts(tn_tr, fp_tr, fn_tr, tp_tr)

        # (3) Validation – sammle p1 und y für Threshold-Suche
        model.eval()
        val_loss = 0.0
        tn_v = fp_v = fn_v = tp_v = 0
        val_p1_all, y_val_all = [], []

        with torch.no_grad():
            for (x704, xext, labels) in val_loader:
                x704   = x704.to(device, non_blocking=True)
                xext   = xext.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                logits = model(x704, xext)
                loss = criterion(logits, labels)
                val_loss += loss.item()

                probs = torch.softmax(logits, dim=1)
                p1 = probs[:, 1].detach().cpu().numpy()
                val_p1_all.append(p1)
                y_val_all.append(labels.detach().cpu().numpy())

                # Argmax-Info (nur fürs Gefühl)
                pred_max = logits.argmax(1)
                cm = confusion_matrix(labels.cpu().numpy(), pred_max.cpu().numpy(), labels=[0, 1])
                if cm.shape == (2,2):
                    tn_v += int(cm[0,0]); fp_v += int(cm[0,1])
                    fn_v += int(cm[1,0]); tp_v += int(cm[1,1])

                del logits, loss, probs, pred_max

        p1_val = np.concatenate(val_p1_all) if len(val_p1_all) else np.array([], dtype=np.float32)
        y_val  = np.concatenate(y_val_all)  if len(y_val_all)  else np.array([], dtype=np.int64)

        # Threshold-Suche
        if thresh_grid is None:
            grid = np.linspace(0.5, 0.99, 100)
        else:
            grid = np.asarray(thresh_grid, dtype=np.float32)

        t_star, tinfo = pick_threshold(
            p1=p1_val, y=y_val, grid=grid,
            max_fp0_rate=fp0_max_rate, objective=thresh_objective
        )

        # Val-Metrik @ t*
        pred_t = (p1_val >= t_star).astype(np.int32)
        cm_t = confusion_matrix(y_val, pred_t, labels=[0, 1])
        if cm_t.shape == (2,2):
            tn_s, fp_s, fn_s, tp_s = int(cm_t[0,0]), int(cm_t[0,1]), int(cm_t[1,0]), int(cm_t[1,1])
        else:
            tn_s = fp_s = fn_s = tp_s = 0
        (p0_s, r0_s, f10_s), (p1_s, r1_s, f11_s), macro_f1_s = metrics_from_counts(tn_s, fp_s, fn_s, tp_s)
        val_acc_s = (tn_s + tp_s) / max(1, (tn_s + fp_s + fn_s + tp_s))
        fp01_val = fp_s  # kritische Fehler (0->1) @t*

        # Logging / Print
        print("\n[VAL @ argmax] (nur Info)")
        (p0_a, r0_a, f10_a), (p1_a, r1_a, f11_a), macro_f1_a = metrics_from_counts(tn_v, fp_v, fn_v, tp_v)
        print(f"  macroF1(argmax)={macro_f1_a:.4f}")

        print(f"[VAL Threshold-Search] fp0_max_rate={fp0_max_rate:.4f} | best t*={t_star:.4f} "
              f"| feasible={tinfo['feasible']} | fpr0={tinfo['fpr0']:.5f}")
        print(f"  Class 0 -> P={p0_s:.4f} | R={r0_s:.4f} | F1={f10_s:.4f}")
        print(f"  Class 1 -> P={p1_s:.4f} | R={r1_s:.4f} | F1={f11_s:.4f}")
        print(f"  macroF1(t*)={macro_f1_s:.4f} | crit.errors 0→1 = {fp01_val}")

        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1}/{epochs} | lr={cur_lr:.6g}")

        # Scheduler
        if epoch + 1 >= use_plateau_after:
            plateau.step(macro_f1_s)

        # Dateien fortschreiben
        with open(os.path.join(save_path, f"{project}_train_loss.txt"), "a") as f:
            f.write(f"{epoch+1}; {running_loss/max(1,num_batches):.4f}\n")
        with open(os.path.join(save_path, f"{project}_train_accuracy.txt"), "a") as f:
            f.write(f"{epoch+1}; {train_acc:.4f}\n")
        with open(os.path.join(save_path, f"{project}_train_f1.txt"), "a") as f:
            f.write(f"{epoch+1}; {macro_f1_t:.4f}\n")

        val_batches = max(1, len(val_loader))
        with open(os.path.join(save_path, f"{project}_val_loss.txt"), "a") as f:
            f.write(f"{epoch+1}; {val_loss/val_batches:.4f}\n")
        with open(os.path.join(save_path, f"{project}_val_accuracy.txt"), "a") as f:
            f.write(f"{epoch+1}; {val_acc_s:.4f}\n")
        with open(os.path.join(save_path, f"{project}_val_f1.txt"), "a") as f:
            f.write(f"{epoch+1}; {macro_f1_s:.4f}\n")

        print(
            f"Epoch {epoch+1}/{epochs}, "
            f"TrainLoss: {running_loss/max(1,num_batches):.4f}, "
            f"TrainAcc: {train_acc:.4f}, TrainF1: {macro_f1_t:.4f}, "
            f"ValLoss: {val_loss/val_batches:.4f}, ValAcc@t*: {val_acc_s:.4f}, ValF1@t*: {macro_f1_s:.4f}"
        )

        if writer is not None:
            with writer.as_default():
                tf.summary.scalar("Loss/train", running_loss/max(1,num_batches), step=epoch)
                tf.summary.scalar("Accuracy/train", train_acc, step=epoch)
                tf.summary.scalar("F1/train_macro", macro_f1_t, step=epoch)

                tf.summary.scalar("Loss/val", val_loss/val_batches, step=epoch)
                tf.summary.scalar("Accuracy/val@t*", val_acc_s, step=epoch)
                tf.summary.scalar("F1/val_macro@t*", macro_f1_s, step=epoch)
                tf.summary.scalar("Val/t_star", t_star, step=epoch)
                tf.summary.scalar("Val/FPR0@t*", tinfo["fpr0"], step=epoch)
                tf.summary.scalar("Val/Class0_P", p0_s, step=epoch)
                tf.summary.scalar("Val/Class0_R", r0_s, step=epoch)
                tf.summary.scalar("Val/Class1_P", p1_s, step=epoch)
                tf.summary.scalar("Val/Class1_R", r1_s, step=epoch)
                writer.flush()

        # Best-Model (nach F1@t*)
        if macro_f1_s > best_f1:
            best_f1 = macro_f1_s
            best_info = {
                "epoch": epoch+1,
                "val_macro_f1_t": float(macro_f1_s),
                "val_t_star": float(t_star),
                "val_fpr0_t": float(tinfo["fpr0"]),
                "objective": thresh_objective
            }
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}  # leichtgewichtig

            best_model_path = os.path.join(save_path, f"{project}_best_model.pth")
            torch.save({"epoch": epoch+1, "state_dict": model.state_dict(), "t_star": float(t_star)}, best_model_path)
            with open(os.path.join(save_path, f"{project}_best_model_info.json"), "w") as jf:
                json.dump({**best_info, "model_save_path": best_model_path}, jf, indent=2)
            print(f"New best model @ epoch {epoch+1}: macroF1(t*)={macro_f1_s:.4f}, t*={t_star:.4f}")

        if not only_save_best_model:
            ep_path = os.path.join(save_path, f"{project}_model_epoch_{epoch+1}_valF1t_{macro_f1_s:.4f}.pth")
            torch.save(model.state_dict(), ep_path)

        # (4) Hard-Negative-Mining für nächste Epoche
        try:
            model.eval()
            train_indices = getattr(dataset_with_cv, "train_idx", list(range(len(dataset_with_cv))))
            neg_indices = [i for i in train_indices if _label_from_pair_idx(i) == 0]
            if len(neg_indices) == 0:
                hard_neg_prev = set()
            else:
                neg_loader = DataLoader(Subset(dataset_with_cv, neg_indices),
                                        batch_size=base_bs, shuffle=False, num_workers=0, drop_last=False)
                scores, buf = [], []
                offset = 0
                with torch.no_grad():
                    for (x704, xext, labels) in neg_loader:
                        x704 = x704.to(device, non_blocking=True)
                        xext = xext.to(device, non_blocking=True)
                        p1 = torch.softmax(model(x704, xext), dim=1)[:, 1].detach().cpu().numpy()
                        n = len(p1)
                        batch_ids = neg_indices[offset:offset+n]
                        offset += n
                        scores.append(p1); buf.extend(batch_ids)
                if len(scores):
                    s = np.concatenate(scores)
                    K = min(int(topk_hard_neg), len(s))
                    topk = np.argpartition(s, -K)[-K:]
                    hard_neg_prev = set(int(buf[i]) for i in topk)
                else:
                    hard_neg_prev = set()
            print(f"[HardNeg] next epoch: {len(hard_neg_prev)} boosted (×{boost_hard_neg})")
        except Exception as e:
            print(f"[HardNeg] skipped: {e}")
            hard_neg_prev = set()

        # Cleanup / Monitoring
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            print(f"[Epoch {epoch+1}] VRAM now: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        if process is not None:
            print(f"[Epoch {epoch+1}] RAM: {process.memory_info().rss/(1024**3):.2f} GB")

    # ---------- Test ----------
    print("\nStart Test Loop")
    # Nutze t* vom besten Modell, wenn vorhanden; sonst letzten t_star aus der Val-Schleife
    if best_info is not None:
        t_test = float(best_info["val_t_star"])
    else:
        # falls Training gar nicht lief
        t_test = 0.5

    model.eval()
    y_true, y_pred, y_p1 = [], [], []

    with torch.no_grad():
        for (x704, xext, labels) in test_loader:
            x704 = x704.to(device, non_blocking=True)
            xext = xext.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            p1 = torch.softmax(model(x704, xext), dim=1)[:, 1]
            pred = (p1 >= t_test).long()

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(pred.cpu().numpy().tolist())
            y_p1.extend(p1.cpu().numpy().tolist())

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro   = f1_score(y_true, y_pred, average="macro", zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TP = np.diag(cm)
    FP = cm.sum(axis=0) - TP
    FN = cm.sum(axis=1) - TP
    TN = cm.sum() - (TP + FP + FN)

    prec, rec, f1_cls, supp = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )

    print(f"Test @ t*={t_test:.4f}")
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"Precision (macro): {prec_macro:.4f}")
    print(f"Recall (macro):    {rec_macro:.4f}")
    print(f"F1-Score (macro):  {f1_macro:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("Class-wise metrics (TEST):")
    print(f"  Class 0 -> P={prec[0]:.4f} | R={rec[0]:.4f} | F1={f1_cls[0]:.4f} | n={supp[0]}")
    print(f"  Class 1 -> P={prec[1]:.4f} | R={rec[1]:.4f} | F1={f1_cls[1]:.4f} | n={supp[1]}")

    fp_class1 = int(cm[0,1])
    print(f"Kritische Fehler (0→1): {fp_class1}")

    # Excel-Export
    conf_df = pd.DataFrame(cm, index=["True_0","True_1"], columns=["Pred_0","Pred_1"])
    per_class = pd.DataFrame({
        "Class":[0,1], "Support":supp, "TP":TP, "FP":FP, "FN":FN, "TN":TN,
        "Precision":prec, "Recall":rec, "F1":f1_cls
    })
    summary = pd.DataFrame({
        "Metric": ["Accuracy", "Precision(macro)", "Recall(macro)", "F1(macro)", "FP_class1(0→1)", "t_star"],
        "Value":  [acc, prec_macro, rec_macro, f1_macro, fp_class1, t_test]
    })
    xlsx_path = os.path.join(save_path, f"{project}_test_metrics.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as xw:
        per_class.to_excel(xw, sheet_name="Per_Class", index=False)
        conf_df.to_excel(xw, sheet_name="Confusion_Matrix")
        summary.to_excel(xw, sheet_name="Summary", index=False)
    print(f"Saved test metrics to {xlsx_path}")

    total_time = time.time() - total_start
    with open(os.path.join(save_path, f"{project}_total_training_time.txt"), "w") as f:
        f.write(f"Total training time: {total_time:.2f} seconds\n")
    print(f"Total training time: {total_time:.2f} s")

    return acc, prec_macro, rec_macro, f1_macro

###### class 1 focus
def nonzero_flat(mask):
    """
    Kompatibel für alte und neue PyTorch-Versionen.
    Gibt 1D-Index-Tensor zurück.
    """
    try:
        return mask.nonzero(as_tuple=True)[0]  # neuere Versionen
    except TypeError:
        # alte Version ohne Keyword-Arg
        return mask.nonzero().view(-1)

def evaluate_cnn_xfeat_thresh(
    model,
    dataset_with_cv,
    train_loader,      # initialer Loader (nur für steps_per_epoch von OneCycle)
    val_loader,
    test_loader,
    epochs=150,
    learning_rate=3e-4,
    device="cpu",
    save_path="./models",
    project="default",
    only_save_best_model=True,
    cv_info=None,
    fold=None,
    # ---- Threshold-Optionen
    fp0_max_rate=0.015,             # max erlaubte 0->1-Fehlrate (FPR0) auf VAL
    thresh_grid=None,               # z.B. np.linspace(0.5, 0.99, 100)
    thresh_objective="f1_class1",    # "macro_f1" | "f1_class1" | "recall_class1"
    # ---- Hard-Mining
    max_negatives_per_epoch=500_000,
    boost_hard_neg= 8.0,
    topk_hard_neg=100_000,
    # >>> Hard-Positives (dynamisch)
    boost_hard_pos=6.0,
    topk_hard_pos=15_000,
    # >>> Hard-Mining (statische Flags)
    use_sc_hard_negs=False,   # same-class in Label 0 boosten
    boost_sc_hard_neg=6.0,    # Boost-Faktor dafür
    sc_hard_neg_max=None,     # ggf. kappen (z.B. 50_000)

    use_perc_hard_pos=False,  # positive Buckets boosten
    perc_hard_pos_values=(35, 40, 45, 50, 55),
    perc_side="scan",         # "scan" | "ref" | "either" | "both"
    boost_perc_hard_pos=3.0,

    # ---- Class-Weights
    class_weights_vec=[1.0, 2.0],

    weight_decay=1e-5,
    grad_accum_steps=4,
    balanced=True,            # <- FIX: richtiger Parametername

    # ---- Recall-Booster für Klasse 1 (FN-Penalty)
    use_fn_penalty=True,
    t_pos=0.70,               # gewünschtes Mindest-p1 für positive Beispiele
    lambda_fn=0.5,            # Stärke der Zusatzstrafe
    t_neg = 0.30,
    lambda_fp = 0.3,
    # NEU:
    recall1_min=0.88,             # z.B. 0.72  -> Mindest-Recall für Klasse 1
    constraint_mode="both",       # "fpr0" | "recall1" | "both" | "none"   #constraint_mode="both", fp0_max_rate=0.006, recall1_min=0.72
        #constraint_mode="recall1", recall1_min=0.72, fp0_max_rate=None
#constraint_mode="fpr0", fp0_max_rate=0.006, recall1_min=None

):
    """
    Trainiert mit Hard-Negatives & Hard-Positives und wählt pro Epoche einen Threshold t*
    auf der Validation (FPR0 <= fp0_max_rate, Ziel=thresh_objective).
    Speichert Best-Model (nach F1@t*) und exportiert Testmetriken inkl. per-Klasse.
    """
    import os, time, json, gc
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
    from sklearn.metrics import (
        confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
        precision_recall_fscore_support
    )

    def _split_xext(xext, dataset):
        """
        Teilt den konkatenierten Nebenkanal in (x44, x27).
        Falls nur ein Teil vorhanden ist, gibt der andere None zurück.
        """
        if xext is None or (hasattr(xext, "numel") and xext.numel() == 0):
            return None, None
        assert xext.dim() == 2, f"erwarte 2D Tensor für xext, bekam {xext.shape}"

        d_grid = int(getattr(dataset, "_grid_dim", 27))  # aus deinem Dataset gesetzt
        d_tot = xext.size(1)
        d44 = max(0, d_tot - d_grid)  # erlaubt auch Fälle ohne Grid oder ohne 44er

        x44 = xext[:, :d44] if d44 > 0 else None
        x27 = xext[:, d44:d44 + d_grid] if d_grid > 0 and d_tot >= d44 + d_grid else None
        return x44, x27

    # optional: RAM-Monitor
    try:
        import psutil
        process = psutil.Process(os.getpid())
    except Exception:
        process = None

    # optional: TensorBoard (nur CPU-Benutzung, kein GPU-Claim)
    try:
        import tensorflow as tf
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass
    except Exception:
        tf = None

    os.makedirs(save_path, exist_ok=True)
    writer = tf.summary.create_file_writer(os.path.join(save_path, "logs")) if tf is not None else None

    # ---------- Hilfsfunktionen ----------
    def _get_y_mapped_from_pair(p):
        # nimmt dein Mapping, falls vorhanden; sonst Fallback
        try:
            return globals()["_y_mapped_from_pair"](p)
        except Exception:
            y = p.get("label", None)
            if y is None or y == 0:
                return None
            return 0 if y == 2 else 1

    def _label_from_pair_idx(i):
        ym = _get_y_mapped_from_pair(dataset_with_cv.all_pairs[i])
        return None if ym is None else int(ym)

    def metrics_from_counts(tn, fp, fn, tp):
        # per Klasse und macro
        prec1 = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec1  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f11   = (2*prec1*rec1)/(prec1+rec1) if (prec1+rec1) > 0 else 0.0

        prec0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        rec0  = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f10   = (2*prec0*rec0)/(prec0+rec0) if (prec0+rec0) > 0 else 0.0

        macro_f1 = 0.5 * (f10 + f11)
        return (prec0, rec0, f10), (prec1, rec1, f11), macro_f1

    def pick_threshold(p1, y, grid,
                       fpr0_max_rate=None,
                       recall1_min=None,
                       constraint_mode="fpr0",
                       objective="macro_f1"):
        """
        Wählt t* aus 'grid' nach 'objective' unter Nebenbedingungen:
          - constraint_mode="fpr0"    -> FPR0 <= fpr0_max_rate
          - constraint_mode="recall1" -> R1   >= recall1_min
          - constraint_mode="both"    -> beide
          - constraint_mode="none"    -> keine Nebenbedingung
        Fallback: Falls keine Schwelle alle Constraints erfüllt,
                  wird diejenige mit minimaler Verletzung gewählt,
                  danach bestes 'objective'.
        """
        import numpy as np
        from sklearn.metrics import confusion_matrix

        if grid is None or len(grid) == 0:
            grid = np.linspace(0.5, 0.99, 100)

        best = None
        best_key = None

        # Für Fallback (minimale Constraint-Verletzung)
        fb_best = None
        fb_key = None

        for t in grid:
            pred = (p1 >= t).astype(np.int32)
            cm = confusion_matrix(y, pred, labels=[0, 1])

            if cm.shape == (2, 2):
                tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
            else:
                tn = fp = fn = tp = 0

            denom = tn + fp
            fpr0 = (fp / denom) if denom > 0 else 0.0
            # Klasse-1 Kennzahlen
            prec1 = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec1 = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f11 = (2 * prec1 * rec1) / (prec1 + rec1) if (prec1 + rec1) > 0 else 0.0

            # Klasse-0 F1 (für macro)
            prec0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
            rec0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            f10 = (2 * prec0 * rec0) / (prec0 + rec0) if (prec0 + rec0) > 0 else 0.0

            macro_f1 = 0.5 * (f10 + f11)

            # Primäre Optimierungsmetrik
            if objective == "f1_class1":
                key_primary = f11
            elif objective == "recall_class1":
                key_primary = rec1
            elif objective == "precision_class1":
                key_primary = prec1
            else:
                key_primary = macro_f1

            # Nebenbedingungen
            feasible = True
            if constraint_mode in ("fpr0", "both") and (fpr0_max_rate is not None):
                feasible = feasible and (fpr0 <= fpr0_max_rate)
            if constraint_mode in ("recall1", "both") and (recall1_min is not None):
                feasible = feasible and (rec1 >= recall1_min)

            # Sortierschlüssel für *feasible* Kandidaten
            #   (1) Feasible, (2) primary hoch, (3) niedriger fpr0, (4) höherer rec1, (5) konservativ höheres t
            key = (1, key_primary, -fpr0, rec1, -t)

            # Für Fallback: Constraintverletzung messen (0 = perfekt)
            v_fpr = max(0.0, (fpr0 - (fpr0_max_rate if fpr0_max_rate is not None else 1.0)))
            v_rec1 = max(0.0, ((recall1_min if recall1_min is not None else 0.0) - rec1))
            # Gewichtung: FPR-Verletzung typ. strenger
            viol = 2.0 * v_fpr + 1.0 * v_rec1
            fb_key_cand = (-viol, key_primary, -fpr0, rec1, -t)

            cand = {
                "t": float(t), "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                "fpr0": float(fpr0), "rec1": float(rec1), "f11": float(f11),
                "prec1": float(prec1), "macro_f1": float(macro_f1),
                "feasible": bool(feasible)
            }

            if feasible:
                if (best_key is None) or (key > best_key):
                    best = cand;
                    best_key = key
            # Fallback-Kandidat immer updaten
            if (fb_key is None) or (fb_key_cand > fb_key):
                fb_best = cand;
                fb_key = fb_key_cand

        if best is not None:
            return best["t"], best
        # Kein t erfüllt alle Constraints -> weichster Verstoß (fb_best)
        return fb_best["t"], fb_best

    # --- Key-Parser / Meta ---
    def _parse_esf_key(key: str):
        # Erwartet Format "..._<inst>_<perc>_<idx>"
        try:
            parts = key.split("_")
            cls = "_".join(parts[:-3]) if len(parts) >= 4 else None
            inst = parts[-3] if len(parts) >= 3 else None
            perc = int(parts[-2]) if len(parts) >= 2 and parts[-2].isdigit() else None
            idx  = int(parts[-1]) if len(parts) >= 1 and parts[-1].isdigit() else None
            return cls, inst, perc, idx
        except Exception:
            return None, None, None, None

    def _pair_meta_from_idx(i):
        p = dataset_with_cv.all_pairs[i]
        k_ref = p.get("esf_ref", "")
        k_scan = p.get("esf_scan", "")
        cls_r, inst_r, perc_r, _ = _parse_esf_key(k_ref)
        cls_s, inst_s, perc_s, _ = _parse_esf_key(k_scan)
        return (cls_r, inst_r, perc_r), (cls_s, inst_s, perc_s)

    def _same_class(i):
        (cr, _, _), (cs, _, _) = _pair_meta_from_idx(i)
        return (cr is not None) and (cs is not None) and (cr == cs)

    def _perc_hit(i, values, side="scan"):
        (cr, ir, pr), (cs, is_, ps) = _pair_meta_from_idx(i)
        V = set(int(v) for v in values)
        if side == "scan":
            return (ps is not None) and (ps in V)
        elif side == "ref":
            return (pr is not None) and (pr in V)
        elif side == "either":
            return ((ps is not None) and (ps in V)) or ((pr is not None) and (pr in V))
        elif side == "both":
            return ((ps is not None) and (ps in V)) and ((pr is not None) and (pr in V))
        else:
            return False

    # --- Sampler: balanced ODER klassisch (mit Boost-Sets) ---
    def make_epoch_sampler_balanced(
        train_idx, all_pairs,
        neg_multiple=2.0,  # pos:neg ~ 1:2
        keep_all_positives=True,
        hard_neg=None, boost_neg=3.0,
        hard_pos=None, boost_pos=8.0
    ):
        hard_neg = set(hard_neg or [])
        hard_pos = set(hard_pos or [])

        pos_idx, neg_idx = [], []
        for i in train_idx:
            y = _get_y_mapped_from_pair(all_pairs[i])
            if y is None:
                continue
            (pos_idx if y == 1 else neg_idx).append(i)

        target_negs = int(len(pos_idx) * float(neg_multiple))
        if len(neg_idx) > target_negs:
            neg_sampled = np.random.choice(neg_idx, size=target_negs, replace=False)
        else:
            neg_sampled = np.asarray(neg_idx, dtype=np.int64)

        indices = np.asarray(pos_idx, dtype=np.int64) if keep_all_positives else np.array([], dtype=np.int64)
        if len(neg_sampled) > 0:
            indices = np.concatenate([indices, neg_sampled])

        weights = np.ones(len(indices), dtype=np.float32)
        for k, idx in enumerate(indices):
            if idx in hard_pos:
                weights[k] = float(boost_pos)
            elif idx in hard_neg:
                weights[k] = float(boost_neg)

        sampler = WeightedRandomSampler(
            weights=torch.tensor(weights, dtype=torch.float32),
            num_samples=len(indices),
            replacement=True
        )
        return sampler, indices

    def make_epoch_sampler(
        train_idx, all_pairs, max_negatives,
        keep_all_positives=True,
        hard_neg=None, boost_neg=5.0,
        hard_pos=None, boost_pos=3.0,
        # NEU:
        hard_neg_sc=None, boost_neg_sc=6.0,
        hard_pos_perc=None, boost_pos_perc=3.0,
    ):
        hard_neg = set(hard_neg or [])
        hard_pos = set(hard_pos or [])
        hard_neg_sc = set(hard_neg_sc or [])
        hard_pos_perc = set(hard_pos_perc or [])

        pos_idx, neg_idx = [], []
        for i in train_idx:
            ym = _get_y_mapped_from_pair(all_pairs[i])
            if ym is None:
                continue
            (pos_idx if ym == 1 else neg_idx).append(i)

        if (max_negatives is not None) and (len(neg_idx) > max_negatives):
            neg_sampled = np.random.choice(neg_idx, size=max_negatives, replace=False)
        else:
            neg_sampled = np.asarray(neg_idx, dtype=np.int64)

        indices = np.asarray(pos_idx, dtype=np.int64) if keep_all_positives else np.array([], dtype=np.int64)
        if len(neg_sampled) > 0:
            indices = np.concatenate([indices, neg_sampled])

        weights = np.ones(len(indices), dtype=np.float32)
        for k, idx in enumerate(indices):
            if idx in hard_neg:       weights[k] *= float(boost_neg)
            if idx in hard_pos:       weights[k] *= float(boost_pos)
            if idx in hard_neg_sc:    weights[k] *= float(boost_neg_sc)
            if idx in hard_pos_perc:  weights[k] *= float(boost_pos_perc)

        sampler = WeightedRandomSampler(
            weights=torch.tensor(weights, dtype=torch.float32),
            num_samples=len(indices),
            replacement=True
        )
        return sampler, indices

    # ---------- Setup ----------
    device = torch.device(device)
    model = model.to(device)

    # Class-Weights
    if class_weights_vec is not None:
        class_weights = torch.tensor(class_weights_vec, dtype=torch.float32, device=device)
    else:
        try:
            class_weights = make_ce_weights_from_dataset(dataset_with_cv, alpha_neg=1.6, device=device)
        except Exception:
            class_weights = torch.tensor([1.0, 1.0], dtype=torch.float32, device=device)
    print("[CE weights]", class_weights.tolist())

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    steps_per_epoch = max(1, len(train_loader))
    onecycle = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=max(learning_rate, 1e-3), epochs=epochs,
        steps_per_epoch=steps_per_epoch, div_factor=25.0, final_div_factor=1e4
    )
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, threshold=1e-4,
        cooldown=1, min_lr=1e-6
    )
    use_plateau_after = max(4, epochs - 4)

    # History
    best_f1 = -1.0
    best_info = None

    hard_neg_prev = set()
    hard_pos_prev = set()
    base_bs = getattr(train_loader, "batch_size", 64)

    # ---------- Test ----------
    print("\nStart Test Loop")
    if best_info is not None:
        t_test = float(best_info["val_t_star"])
    else:
        t_test = 0.5

    os.makedirs(save_path, exist_ok=True)

    model.eval()
    y_true, y_pred, y_p1 = [], [], []

    # fortlaufende Indextabelle für Test-Items (Mapping zurück auf all_pairs)
    test_indices_seq = list(getattr(dataset_with_cv, "test_idx", list(range(len(dataset_with_cv)))))
    ptr = 0
    per_item_indices = []

    with torch.no_grad():
        for (x704, xext, labels) in test_loader:
            x704 = x704.to(device, non_blocking=True)
            xext = xext.to(device, non_blocking=True)

            labels = labels.to(device, non_blocking=True)

            x44, x27 = _split_xext(xext, dataset_with_cv)
            if x44 is not None: x44 = x44.to(device, non_blocking=True)
            if x27 is not None: x27 = x27.to(device, non_blocking=True)
            #print("extra: mean", x44.mean().item(), "std", x44.std().item(),
                  #"nonzero%", (x44 != 0).float().mean().item() * 100)
            #print("grid: mean", x27.mean().item(), "std", x27.std().item(),
                  #"nonzero%", (x27 != 0).float().mean().item() * 100)

            logits = model(x704, x44, x27)
            p1 = torch.softmax(logits, dim=1)[:, 1] # <-- TENSOR
            pred = (p1 >= t_test).to(torch.long)  # <-- TENSOR
            # Für Statistik/Logging:
            p1_np = p1.detach().cpu().numpy()  # Tensor auf device
            print("p1 mean:", p1_np.mean(), "min:", p1_np.min(), "max:", p1.max())
            # y_true/y_pred sammeln:
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(pred.cpu().numpy().tolist())
            y_p1.extend(p1_np.tolist())

            bs = labels.size(0)
            per_item_indices.extend(test_indices_seq[ptr:ptr + bs])
            ptr += bs

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(pred.cpu().numpy().tolist())
            y_p1.extend(p1.cpu().numpy().tolist())

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro   = f1_score(y_true, y_pred, average="macro", zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TP = np.diag(cm)
    FP = cm.sum(axis=0) - TP
    FN = cm.sum(axis=1) - TP
    TN = cm.sum() - (TP + FP + FN)

    prec, rec, f1_cls, supp = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )

    print(f"Test @ t*={t_test:.4f}")
    print(f"Accuracy: {acc * 100:.2f}%")
    print(f"Precision (macro): {prec_macro:.4f}")
    print(f"Recall (macro):    {rec_macro:.4f}")
    print(f"F1-Score (macro):  {f1_macro:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("Class-wise metrics (TEST):")
    print(f"  Class 0 -> P={prec[0]:.4f} | R={rec[0]:.4f} | F1={f1_cls[0]:.4f} | n={supp[0]}")
    print(f"  Class 1 -> P={prec[1]:.4f} | R={rec[1]:.4f} | F1={f1_cls[1]:.4f} | n={supp[1]}")

    fp_class1 = int(cm[0, 1])
    print(f"Kritische Fehler (0→1): {fp_class1}")

    # ---------- FP-JSON & Aggregationen ----------
    def _parse_key(key):
        p = key.split("_")
        return "_".join(p[:-3]), p[-3], p[-2], int(p[-1])

    def _inc(d, k):
        d[k] = d.get(k, 0) + 1

    def _norm(x):
        return "NA" if x is None else str(x)

    # Sanity: per_item_indices absichern
    if len(per_item_indices) != len(y_true):
        from torch.utils.data import Subset
        ds = getattr(test_loader, "dataset", None)
        if isinstance(ds, Subset):
            candidate = list(map(int, ds.indices))
        elif hasattr(dataset_with_cv, "test_idx"):
            candidate = list(map(int, dataset_with_cv.test_idx))
        else:
            candidate = list(range(len(y_true)))
        per_item_indices = candidate[:len(y_true)]
    assert len(per_item_indices) == len(y_true), "index mapping mismatch"

    fp_list_cls1, fp_list_cls0 = [], []
    by_cls_pair_01, by_inst_pair_01, by_perc_pair_01 = {}, {}, {}
    by_cls_pair_10, by_inst_pair_10, by_perc_pair_10 = {}, {}, {}

    for (yt, yp, p1_val, ds_idx) in zip(y_true, y_pred, y_p1, per_item_indices):
        pair = dataset_with_cv.all_pairs[ds_idx]
        k_ref = pair.get("esf_ref", "")
        k_scan = pair.get("esf_scan", "")
        try:
            cls_r, inst_r, perc_r, idx_r = _parse_key(k_ref)
            cls_s, inst_s, perc_s, idx_s = _parse_key(k_scan)
        except Exception:
            cls_r = inst_r = perc_r = idx_r = cls_s = inst_s = perc_s = idx_s = None

        entry = {
            "ds_index": int(ds_idx),
            "true": int(yt),
            "pred": int(yp),
            "p1": float(p1_val),
            "threshold": float(t_test),
            "esf_ref": k_ref,
            "esf_scan": k_scan,
            "parsed_ref": {"cls": cls_r, "inst": inst_r, "perc": perc_r, "idx": idx_r},
            "parsed_scan": {"cls": cls_s, "inst": inst_s, "perc": perc_s, "idx": idx_s},
        }

        _cls_pair = f"{_norm(cls_r)}|{_norm(cls_s)}"
        _inst_pair = f"{_norm(inst_r)}|{_norm(inst_s)}"
        _perc_pair = f"{_norm(perc_r)}|{_norm(perc_s)}"

        if yt == 0 and yp == 1:
            fp_list_cls1.append(entry)
            _inc(by_cls_pair_01, _cls_pair)
            _inc(by_inst_pair_01, _inst_pair)
            _inc(by_perc_pair_01, _perc_pair)
        elif yt == 1 and yp == 0:
            fp_list_cls0.append(entry)
            _inc(by_cls_pair_10, _cls_pair)
            _inc(by_inst_pair_10, _inst_pair)
            _inc(by_perc_pair_10, _perc_pair)

    def _sorted_items(d):
        return [{"pair": k, "count": v} for k, v in sorted(d.items(), key=lambda kv: kv[1], reverse=True)]

    fp_json = {
        "project": project,
        "threshold": float(t_test),
        "counts": {
            "fp_class1_0to1": len(fp_list_cls1),
            "fp_class0_1to0": len(fp_list_cls0)
        },
        "aggregation_by_perc": {
            "class1_0to1": {k: v for k, v in by_perc_pair_01.items()},
            "class0_1to0": {k: v for k, v in by_perc_pair_10.items()}
        },
        "aggregation_by_pairs": {
            "0to1": {
                "by_cls_pair_hist": by_cls_pair_01,
                "by_inst_pair_hist": by_inst_pair_01,
                "by_perc_pair_hist": by_perc_pair_01,
                "top_by_cls_pair": _sorted_items(by_cls_pair_01),
                "top_by_inst_pair": _sorted_items(by_inst_pair_01),
                "top_by_perc_pair": _sorted_items(by_perc_pair_01),
            },
            "1to0": {
                "by_cls_pair_hist": by_cls_pair_10,
                "by_inst_pair_hist": by_inst_pair_10,
                "by_perc_pair_hist": by_perc_pair_10,
                "top_by_cls_pair": _sorted_items(by_cls_pair_10),
                "top_by_inst_pair": _sorted_items(by_inst_pair_10),
                "top_by_perc_pair": _sorted_items(by_perc_pair_10),
            }
        },
        "fp_details": {
            "class1_0to1": fp_list_cls1,
            "class0_1to0": fp_list_cls0
        }
    }

    # --- ROBUST SAVE (mit Fehlerausgabe) ---
    fp_json_path = os.path.join(save_path, f"{project}_test_false_positives.json")
    try:
        with open(fp_json_path, "w", encoding="utf-8") as jf:
            json.dump(fp_json, jf, indent=2, ensure_ascii=False)
        print(f"Saved FP details to {fp_json_path}")
    except Exception as e:
        print(f"[SAVE-ERROR] FP JSON: {e}")

    # Excel-Export
    conf_df = pd.DataFrame(cm, index=["True_0", "True_1"], columns=["Pred_0", "Pred_1"])
    per_class = pd.DataFrame({
        "Class": [0, 1], "Support": supp, "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "Precision": prec, "Recall": rec, "F1": f1_cls
    })
    summary = pd.DataFrame({
        "Metric": ["Accuracy", "Precision(macro)", "Recall(macro)", "F1(macro)", "FP_class1(0→1)", "t_star"],
        "Value": [acc, prec_macro, rec_macro, f1_macro, fp_class1, t_test]
    })
    xlsx_path = os.path.join(save_path, f"{project}_test_metrics.xlsx")
    try:
        with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as xw:
            per_class.to_excel(xw, sheet_name="Per_Class", index=False)
            conf_df.to_excel(xw, sheet_name="Confusion_Matrix")
            summary.to_excel(xw, sheet_name="Summary", index=False)
        print(f"Saved test metrics to {xlsx_path}")
    except Exception as e:
        print(f"[SAVE-ERROR] Excel: {e}")

    total_time = time.time() - total_start
    tt_path = os.path.join(save_path, f"{project}_total_training_time.txt")
    try:
        with open(tt_path, "w") as f:
            f.write(f"Total training time: {total_time:.2f} seconds\n")
        print(f"Saved total training time to {tt_path}")
    except Exception as e:
        print(f"[SAVE-ERROR] total_training_time: {e}")
    print(f"Total training time: {total_time:.2f} s")

    return acc, prec_macro, rec_macro, f1_macro


def train_and_evaluate_cnn_xfeat_thresh(
    model,
    dataset_with_cv,
    train_loader,      # initialer Loader (nur für steps_per_epoch von OneCycle)
    val_loader,
    test_loader,
    epochs=150,
    learning_rate=3e-4,
    device="cpu",
    save_path="./models",
    project="default",
    only_save_best_model=True,
    cv_info=None,
    fold=None,
    # ---- Threshold-Optionen
    fp0_max_rate=0.015,             # max erlaubte 0->1-Fehlrate (FPR0) auf VAL
    thresh_grid=None,               # z.B. np.linspace(0.5, 0.99, 100)
    thresh_objective="f1_class1",    # "macro_f1" | "f1_class1" | "recall_class1"
    # ---- Hard-Mining
    max_negatives_per_epoch=500_000,
    boost_hard_neg= 8.0,
    topk_hard_neg=100_000,
    # >>> Hard-Positives (dynamisch)
    boost_hard_pos=6.0,
    topk_hard_pos=15_000,
    # >>> Hard-Mining (statische Flags)
    use_sc_hard_negs=False,   # same-class in Label 0 boosten
    boost_sc_hard_neg=6.0,    # Boost-Faktor dafür
    sc_hard_neg_max=None,     # ggf. kappen (z.B. 50_000)

    use_perc_hard_pos=False,  # positive Buckets boosten
    perc_hard_pos_values=(35, 40, 45, 50, 55),
    perc_side="scan",         # "scan" | "ref" | "either" | "both"
    boost_perc_hard_pos=3.0,

    # ---- Class-Weights
    class_weights_vec=[1.0, 2.0],

    weight_decay=1e-5,
    grad_accum_steps=4,
    balanced=True,            # <- FIX: richtiger Parametername

    # ---- Recall-Booster für Klasse 1 (FN-Penalty)
    use_fn_penalty=True,
    t_pos=0.70,               # gewünschtes Mindest-p1 für positive Beispiele
    lambda_fn=0.5,            # Stärke der Zusatzstrafe
    t_neg = 0.30,
    lambda_fp = 0.3,
    # NEU:
    recall1_min=0.88,             # z.B. 0.72  -> Mindest-Recall für Klasse 1
    constraint_mode="both",       # "fpr0" | "recall1" | "both" | "none"   #constraint_mode="both", fp0_max_rate=0.006, recall1_min=0.72
        #constraint_mode="recall1", recall1_min=0.72, fp0_max_rate=None
#constraint_mode="fpr0", fp0_max_rate=0.006, recall1_min=None

):
    """
    Trainiert mit Hard-Negatives & Hard-Positives und wählt pro Epoche einen Threshold t*
    auf der Validation (FPR0 <= fp0_max_rate, Ziel=thresh_objective).
    Speichert Best-Model (nach F1@t*) und exportiert Testmetriken inkl. per-Klasse.
    """
    import os, time, json, gc
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
    from sklearn.metrics import (
        confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
        precision_recall_fscore_support
    )

    def _split_xext(xext, dataset):
        """
        Teilt den konkatenierten Nebenkanal in (x44, x27).
        Falls nur ein Teil vorhanden ist, gibt der andere None zurück.
        """
        if xext is None or (hasattr(xext, "numel") and xext.numel() == 0):
            return None, None
        assert xext.dim() == 2, f"erwarte 2D Tensor für xext, bekam {xext.shape}"

        d_grid = int(getattr(dataset, "_grid_dim", 27))  # aus deinem Dataset gesetzt
        d_tot = xext.size(1)
        d44 = max(0, d_tot - d_grid)  # erlaubt auch Fälle ohne Grid oder ohne 44er

        x44 = xext[:, :d44] if d44 > 0 else None
        x27 = xext[:, d44:d44 + d_grid] if d_grid > 0 and d_tot >= d44 + d_grid else None
        return x44, x27

    # optional: RAM-Monitor
    try:
        import psutil
        process = psutil.Process(os.getpid())
    except Exception:
        process = None

    # optional: TensorBoard (nur CPU-Benutzung, kein GPU-Claim)
    try:
        import tensorflow as tf
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass
    except Exception:
        tf = None

    os.makedirs(save_path, exist_ok=True)
    writer = tf.summary.create_file_writer(os.path.join(save_path, "logs")) if tf is not None else None

    # ---------- Hilfsfunktionen ----------
    def _get_y_mapped_from_pair(p):
        # nimmt dein Mapping, falls vorhanden; sonst Fallback
        try:
            return globals()["_y_mapped_from_pair"](p)
        except Exception:
            y = p.get("label", None)
            if y is None or y == 0:
                return None
            return 0 if y == 2 else 1

    def _label_from_pair_idx(i):
        ym = _get_y_mapped_from_pair(dataset_with_cv.all_pairs[i])
        return None if ym is None else int(ym)

    def metrics_from_counts(tn, fp, fn, tp):
        # per Klasse und macro
        prec1 = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec1  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f11   = (2*prec1*rec1)/(prec1+rec1) if (prec1+rec1) > 0 else 0.0

        prec0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        rec0  = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f10   = (2*prec0*rec0)/(prec0+rec0) if (prec0+rec0) > 0 else 0.0

        macro_f1 = 0.5 * (f10 + f11)
        return (prec0, rec0, f10), (prec1, rec1, f11), macro_f1

    def pick_threshold(p1, y, grid,
                       fpr0_max_rate=None,
                       recall1_min=None,
                       constraint_mode="fpr0",
                       objective="macro_f1"):
        """
        Wählt t* aus 'grid' nach 'objective' unter Nebenbedingungen:
          - constraint_mode="fpr0"    -> FPR0 <= fpr0_max_rate
          - constraint_mode="recall1" -> R1   >= recall1_min
          - constraint_mode="both"    -> beide
          - constraint_mode="none"    -> keine Nebenbedingung
        Fallback: Falls keine Schwelle alle Constraints erfüllt,
                  wird diejenige mit minimaler Verletzung gewählt,
                  danach bestes 'objective'.
        """
        import numpy as np
        from sklearn.metrics import confusion_matrix

        if grid is None or len(grid) == 0:
            grid = np.linspace(0.5, 0.99, 100)

        best = None
        best_key = None

        # Für Fallback (minimale Constraint-Verletzung)
        fb_best = None
        fb_key = None

        for t in grid:
            pred = (p1 >= t).astype(np.int32)
            cm = confusion_matrix(y, pred, labels=[0, 1])

            if cm.shape == (2, 2):
                tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
            else:
                tn = fp = fn = tp = 0

            denom = tn + fp
            fpr0 = (fp / denom) if denom > 0 else 0.0
            # Klasse-1 Kennzahlen
            prec1 = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec1 = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f11 = (2 * prec1 * rec1) / (prec1 + rec1) if (prec1 + rec1) > 0 else 0.0

            # Klasse-0 F1 (für macro)
            prec0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
            rec0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            f10 = (2 * prec0 * rec0) / (prec0 + rec0) if (prec0 + rec0) > 0 else 0.0

            macro_f1 = 0.5 * (f10 + f11)

            # Primäre Optimierungsmetrik
            if objective == "f1_class1":
                key_primary = f11
            elif objective == "recall_class1":
                key_primary = rec1
            elif objective == "precision_class1":
                key_primary = prec1
            else:
                key_primary = macro_f1

            # Nebenbedingungen
            feasible = True
            if constraint_mode in ("fpr0", "both") and (fpr0_max_rate is not None):
                feasible = feasible and (fpr0 <= fpr0_max_rate)
            if constraint_mode in ("recall1", "both") and (recall1_min is not None):
                feasible = feasible and (rec1 >= recall1_min)

            # Sortierschlüssel für *feasible* Kandidaten
            #   (1) Feasible, (2) primary hoch, (3) niedriger fpr0, (4) höherer rec1, (5) konservativ höheres t
            key = (1, key_primary, -fpr0, rec1, -t)

            # Für Fallback: Constraintverletzung messen (0 = perfekt)
            v_fpr = max(0.0, (fpr0 - (fpr0_max_rate if fpr0_max_rate is not None else 1.0)))
            v_rec1 = max(0.0, ((recall1_min if recall1_min is not None else 0.0) - rec1))
            # Gewichtung: FPR-Verletzung typ. strenger
            viol = 2.0 * v_fpr + 1.0 * v_rec1
            fb_key_cand = (-viol, key_primary, -fpr0, rec1, -t)

            cand = {
                "t": float(t), "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                "fpr0": float(fpr0), "rec1": float(rec1), "f11": float(f11),
                "prec1": float(prec1), "macro_f1": float(macro_f1),
                "feasible": bool(feasible)
            }

            if feasible:
                if (best_key is None) or (key > best_key):
                    best = cand;
                    best_key = key
            # Fallback-Kandidat immer updaten
            if (fb_key is None) or (fb_key_cand > fb_key):
                fb_best = cand;
                fb_key = fb_key_cand

        if best is not None:
            return best["t"], best
        # Kein t erfüllt alle Constraints -> weichster Verstoß (fb_best)
        return fb_best["t"], fb_best

    # --- Key-Parser / Meta ---
    def _parse_esf_key(key: str):
        # Erwartet Format "..._<inst>_<perc>_<idx>"
        try:
            parts = key.split("_")
            cls = "_".join(parts[:-3]) if len(parts) >= 4 else None
            inst = parts[-3] if len(parts) >= 3 else None
            perc = int(parts[-2]) if len(parts) >= 2 and parts[-2].isdigit() else None
            idx  = int(parts[-1]) if len(parts) >= 1 and parts[-1].isdigit() else None
            return cls, inst, perc, idx
        except Exception:
            return None, None, None, None

    def _pair_meta_from_idx(i):
        p = dataset_with_cv.all_pairs[i]
        k_ref = p.get("esf_ref", "")
        k_scan = p.get("esf_scan", "")
        cls_r, inst_r, perc_r, _ = _parse_esf_key(k_ref)
        cls_s, inst_s, perc_s, _ = _parse_esf_key(k_scan)
        return (cls_r, inst_r, perc_r), (cls_s, inst_s, perc_s)

    def _same_class(i):
        (cr, _, _), (cs, _, _) = _pair_meta_from_idx(i)
        return (cr is not None) and (cs is not None) and (cr == cs)

    def _perc_hit(i, values, side="scan"):
        (cr, ir, pr), (cs, is_, ps) = _pair_meta_from_idx(i)
        V = set(int(v) for v in values)
        if side == "scan":
            return (ps is not None) and (ps in V)
        elif side == "ref":
            return (pr is not None) and (pr in V)
        elif side == "either":
            return ((ps is not None) and (ps in V)) or ((pr is not None) and (pr in V))
        elif side == "both":
            return ((ps is not None) and (ps in V)) and ((pr is not None) and (pr in V))
        else:
            return False

    # --- Sampler: balanced ODER klassisch (mit Boost-Sets) ---
    def make_epoch_sampler_balanced(
        train_idx, all_pairs,
        neg_multiple=2.0,  # pos:neg ~ 1:2
        keep_all_positives=True,
        hard_neg=None, boost_neg=3.0,
        hard_pos=None, boost_pos=8.0
    ):
        hard_neg = set(hard_neg or [])
        hard_pos = set(hard_pos or [])

        pos_idx, neg_idx = [], []
        for i in train_idx:
            y = _get_y_mapped_from_pair(all_pairs[i])
            if y is None:
                continue
            (pos_idx if y == 1 else neg_idx).append(i)

        target_negs = int(len(pos_idx) * float(neg_multiple))
        if len(neg_idx) > target_negs:
            neg_sampled = np.random.choice(neg_idx, size=target_negs, replace=False)
        else:
            neg_sampled = np.asarray(neg_idx, dtype=np.int64)

        indices = np.asarray(pos_idx, dtype=np.int64) if keep_all_positives else np.array([], dtype=np.int64)
        if len(neg_sampled) > 0:
            indices = np.concatenate([indices, neg_sampled])

        weights = np.ones(len(indices), dtype=np.float32)
        for k, idx in enumerate(indices):
            if idx in hard_pos:
                weights[k] = float(boost_pos)
            elif idx in hard_neg:
                weights[k] = float(boost_neg)

        sampler = WeightedRandomSampler(
            weights=torch.tensor(weights, dtype=torch.float32),
            num_samples=len(indices),
            replacement=True
        )
        return sampler, indices

    def make_epoch_sampler(
        train_idx, all_pairs, max_negatives,
        keep_all_positives=True,
        hard_neg=None, boost_neg=5.0,
        hard_pos=None, boost_pos=3.0,
        # NEU:
        hard_neg_sc=None, boost_neg_sc=6.0,
        hard_pos_perc=None, boost_pos_perc=3.0,
    ):
        hard_neg = set(hard_neg or [])
        hard_pos = set(hard_pos or [])
        hard_neg_sc = set(hard_neg_sc or [])
        hard_pos_perc = set(hard_pos_perc or [])

        pos_idx, neg_idx = [], []
        for i in train_idx:
            ym = _get_y_mapped_from_pair(all_pairs[i])
            if ym is None:
                continue
            (pos_idx if ym == 1 else neg_idx).append(i)

        if (max_negatives is not None) and (len(neg_idx) > max_negatives):
            neg_sampled = np.random.choice(neg_idx, size=max_negatives, replace=False)
        else:
            neg_sampled = np.asarray(neg_idx, dtype=np.int64)

        indices = np.asarray(pos_idx, dtype=np.int64) if keep_all_positives else np.array([], dtype=np.int64)
        if len(neg_sampled) > 0:
            indices = np.concatenate([indices, neg_sampled])

        weights = np.ones(len(indices), dtype=np.float32)
        for k, idx in enumerate(indices):
            if idx in hard_neg:       weights[k] *= float(boost_neg)
            if idx in hard_pos:       weights[k] *= float(boost_pos)
            if idx in hard_neg_sc:    weights[k] *= float(boost_neg_sc)
            if idx in hard_pos_perc:  weights[k] *= float(boost_pos_perc)

        sampler = WeightedRandomSampler(
            weights=torch.tensor(weights, dtype=torch.float32),
            num_samples=len(indices),
            replacement=True
        )
        return sampler, indices

    # ---------- Setup ----------
    device = torch.device(device)
    model = model.to(device)

    # Class-Weights
    if class_weights_vec is not None:
        class_weights = torch.tensor(class_weights_vec, dtype=torch.float32, device=device)
    else:
        try:
            class_weights = make_ce_weights_from_dataset(dataset_with_cv, alpha_neg=1.6, device=device)
        except Exception:
            class_weights = torch.tensor([1.0, 1.0], dtype=torch.float32, device=device)
    print("[CE weights]", class_weights.tolist())

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    steps_per_epoch = max(1, len(train_loader))
    onecycle = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=max(learning_rate, 1e-3), epochs=epochs,
        steps_per_epoch=steps_per_epoch, div_factor=25.0, final_div_factor=1e4
    )
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, threshold=1e-4,
        cooldown=1, min_lr=1e-6
    )
    use_plateau_after = max(4, epochs - 4)

    # History
    best_f1 = -1.0
    best_info = None

    hard_neg_prev = set()
    hard_pos_prev = set()
    base_bs = getattr(train_loader, "batch_size", 64)

    # ---------- Training ----------
    total_start = time.time()

    # ---- Static Sets (kein Model-Forward nötig)
    train_indices_all = getattr(dataset_with_cv, "train_idx", list(range(len(dataset_with_cv))))

    static_sc_hard_neg = set()
    if use_sc_hard_negs:
        for i in train_indices_all:
            y = _label_from_pair_idx(i)
            if y == 0 and _same_class(i):
                static_sc_hard_neg.add(i)
        if sc_hard_neg_max is not None and len(static_sc_hard_neg) > sc_hard_neg_max:
            static_sc_hard_neg = set(np.random.choice(list(static_sc_hard_neg), size=int(sc_hard_neg_max), replace=False))

    static_perc_hard_pos = set()
    if use_perc_hard_pos:
        for i in train_indices_all:
            y = _label_from_pair_idx(i)
            if y == 1 and _perc_hit(i, perc_hard_pos_values, side=perc_side):
                static_perc_hard_pos.add(i)

    print(f"[StaticHard] same-class neg: {len(static_sc_hard_neg)} | perc-hard pos: {len(static_perc_hard_pos)} ({perc_side} in {list(perc_hard_pos_values)})")

    for epoch in range(epochs):
        # (1) Epoche: Sampler bauen
        try:
            if balanced:
                sampler, ep_indices = make_epoch_sampler_balanced(
                    dataset_with_cv.train_idx,
                    dataset_with_cv.all_pairs,
                    neg_multiple=2.0,  # 1.5–2.5 testen
                    keep_all_positives=True,
                    hard_neg=hard_neg_prev, boost_neg=3.0,
                    hard_pos=hard_pos_prev, boost_pos=8.0
                )
            else:
                sampler, ep_indices = make_epoch_sampler(
                    dataset_with_cv.train_idx,
                    dataset_with_cv.all_pairs,
                    max_negatives=max_negatives_per_epoch,
                    keep_all_positives=True,
                    hard_neg=hard_neg_prev,  boost_neg=boost_hard_neg,
                    hard_pos=hard_pos_prev,  boost_pos=boost_hard_pos,
                    hard_neg_sc=static_sc_hard_neg,    boost_neg_sc=boost_sc_hard_neg,
                    hard_pos_perc=static_perc_hard_pos, boost_pos_perc=boost_perc_hard_pos
                )

            tr_loader = DataLoader(
                Subset(dataset_with_cv, ep_indices),
                batch_size=base_bs, sampler=sampler, num_workers=0, drop_last=False
            )

            print(f"[HardMining] next epoch: "
                  f"dyn_hard_neg(topk)={len(hard_neg_prev)} (×{boost_hard_neg}), "
                  f"dyn_hard_pos(loss)={len(hard_pos_prev)} (×{boost_hard_pos}), "
                  f"static_sameclass_neg={len(static_sc_hard_neg)} (×{boost_sc_hard_neg}), "
                  f"static_perc_pos={len(static_perc_hard_pos)} (×{boost_perc_hard_pos})")
        except Exception as e:
            print(f"[Sampler-Fallback] {e}")
            tr_loader = train_loader  # initialer

        # (2) Train
        model.train()
        running_loss = 0.0
        tn_tr = fp_tr = fn_tr = tp_tr = 0
        optimizer.zero_grad(set_to_none=True)
        accum = 0
        num_batches = max(1, len(tr_loader))

        for b, (x704, xext, labels) in enumerate(tr_loader, start=1):
            x704   = x704.to(device, non_blocking=True)
            xext   = xext.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            x44, x27 = _split_xext(xext, dataset_with_cv)
            logits = model(x704, x44, x27)

            # Basis-CE
            ce = criterion(logits, labels)

            # Recall-Booster Klasse 1 (FN-Penalty)
            if use_fn_penalty:
                p1 = torch.softmax(logits, dim=1)[:, 1]  # mit Grad
                pos_mask = (labels == 1)
                neg_mask = (labels == 0)

                # Recall-Booster (halte Positiven-Score oben)
                pen_fn = torch.relu(t_pos - p1)[pos_mask].mean() if pos_mask.any() else logits.new_zeros(())

                # Precision-Booster (drücke p1 bei Negativen)
                pen_fp = torch.relu(p1 - t_neg)[neg_mask].mean() if neg_mask.any() else logits.new_zeros(())

                loss = ce + lambda_fn * pen_fn + lambda_fp * pen_fp
            else:
                loss = ce

            # Accumulation
            loss = loss / max(1, grad_accum_steps)
            loss.backward()
            accum += 1

            # Optimizer-Step
            if (accum == grad_accum_steps) or (b == num_batches):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                onecycle.step()
                accum = 0

            running_loss += loss.item() * max(1, grad_accum_steps)

            # Online-Counts (Argmax)
            with torch.no_grad():
                pred = logits.argmax(1)
                cm = confusion_matrix(labels.detach().cpu().numpy(),
                                      pred.detach().cpu().numpy(), labels=[0, 1])
                if cm.shape == (2,2):
                    tn_tr += int(cm[0,0]); fp_tr += int(cm[0,1])
                    fn_tr += int(cm[1,0]); tp_tr += int(cm[1,1])

            del logits, loss, ce
            if torch.cuda.is_available() and (b % 100 == 0):
                torch.cuda.empty_cache()

        train_total = tn_tr + fp_tr + fn_tr + tp_tr
        train_acc = (tn_tr + tp_tr) / max(1, train_total)
        (p0_t, r0_t, f10_t), (p1_t, r1_t, f11_t), macro_f1_t = metrics_from_counts(tn_tr, fp_tr, fn_tr, tp_tr)

        # (3) Validation – sammle p1 und y für Threshold-Suche
        model.eval()
        val_loss = 0.0
        tn_v = fp_v = fn_v = tp_v = 0
        val_p1_all, y_val_all = [], []

        with torch.no_grad():
            for (x704, xext, labels) in val_loader:
                x704   = x704.to(device, non_blocking=True)
                xext   = xext.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                #print("xext: ", xext)

                x44, x27 = _split_xext(xext, dataset_with_cv)

                #print("extra: mean", x44.mean().item(), "std", x44.std().item(),
                #      "nonzero%", (x44 != 0).float().mean().item() * 100)
                #print("grid: mean", x27.mean().item(), "std", x27.std().item(),
                #      "nonzero%", (x27 != 0).float().mean().item() * 100)

                logits = model(x704, x44, x27)
                loss = criterion(logits, labels)
                val_loss += loss.item()

                probs = torch.softmax(logits, dim=1)
                p1 = probs[:, 1].detach().cpu().numpy()
                #print("p1 mean:", p1.mean(), "min:", p1.min(), "max:", p1.max())
                val_p1_all.append(p1)
                y_val_all.append(labels.detach().cpu().numpy())

                pred_max = logits.argmax(1)
                cm = confusion_matrix(labels.cpu().numpy(), pred_max.cpu().numpy(), labels=[0, 1])
                if cm.shape == (2,2):
                    tn_v += int(cm[0,0]); fp_v += int(cm[0,1])
                    fn_v += int(cm[1,0]); tp_v += int(cm[1,1])

                del logits, loss, probs, pred_max

        p1_val = np.concatenate(val_p1_all) if len(val_p1_all) else np.array([], dtype=np.float32)
        y_val  = np.concatenate(y_val_all)  if len(y_val_all)  else np.array([], dtype=np.int64)

        # Threshold-Suche
        if thresh_grid is None:
            grid = np.linspace(0.5, 0.99, 100)
        else:
            grid = np.asarray(thresh_grid, dtype=np.float32)

        t_star, tinfo = pick_threshold(
            p1=p1_val, y=y_val, grid=grid,
            fpr0_max_rate=fp0_max_rate,
            recall1_min=recall1_min,
            constraint_mode=constraint_mode,
            objective=thresh_objective
        )


        # Val-Metrik @ t*
        pred_t = (p1_val >= t_star).astype(np.int32)
        cm_t = confusion_matrix(y_val, pred_t, labels=[0, 1])
        if cm_t.shape == (2,2):
            tn_s, fp_s, fn_s, tp_s = int(cm_t[0,0]), int(cm_t[0,1]), int(cm_t[1,0]), int(cm_t[1,1])
        else:
            tn_s = fp_s = fn_s = tp_s = 0
        (p0_s, r0_s, f10_s), (p1_s, r1_s, f11_s), macro_f1_s = metrics_from_counts(tn_s, fp_s, fn_s, tp_s)
        val_acc_s = (tn_s + tp_s) / max(1, (tn_s + fp_s + fn_s + tp_s))
        fp01_val = fp_s

        print("\n[VAL @ argmax] (nur Info)")
        (p0_a, r0_a, f10_a), (p1_a, r1_a, f11_a), macro_f1_a = metrics_from_counts(tn_v, fp_v, fn_v, tp_v)
        print(f"  macroF1(argmax)={macro_f1_a:.4f}")

        print(f"[VAL Threshold-Search] mode={constraint_mode} | t*={t_star:.4f} "
              f"| feasible={tinfo['feasible']} | fpr0={tinfo['fpr0']:.5f} | rec1={tinfo['rec1']:.4f}")

        print(f"  Class 0 -> P={p0_s:.4f} | R={r0_s:.4f} | F1={f10_s:.4f}")
        print(f"  Class 1 -> P={p1_s:.4f} | R={r1_s:.4f} | F1={f11_s:.4f}")
        print(f"  macroF1(t*)={macro_f1_s:.4f} | crit.errors 0→1 = {fp01_val}")

        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1}/{epochs} | lr={cur_lr:.6g}")

        if epoch + 1 >= use_plateau_after:
            plateau.step(macro_f1_s)

        # Dateien fortschreiben (einheitlich mit Epoche)
        with open(os.path.join(save_path, f"{project}_train_loss.txt"), "a") as f:
            f.write(f"{epoch+1}; {running_loss/max(1,num_batches):.4f}\n")
        with open(os.path.join(save_path, f"{project}_train_accuracy.txt"), "a") as f:
            f.write(f"{epoch+1}; {train_acc:.4f}\n")
        with open(os.path.join(save_path, f"{project}_train_f1.txt"), "a") as f:
            f.write(f"{epoch+1}; {macro_f1_t:.4f}\n")

        val_batches = max(1, len(val_loader))
        with open(os.path.join(save_path, f"{project}_val_loss.txt"), "a") as f:
            f.write(f"{epoch+1}; {val_loss/val_batches:.4f}\n")
        with open(os.path.join(save_path, f"{project}_val_accuracy.txt"), "a") as f:
            f.write(f"{epoch+1}; {val_acc_s:.4f}\n")
        with open(os.path.join(save_path, f"{project}_val_f1.txt"), "a") as f:
            f.write(f"{epoch+1}; {macro_f1_s:.4f}\n")

        print(
            f"Epoch {epoch+1}/{epochs}, "
            f"TrainLoss: {running_loss/max(1,num_batches):.4f}, "
            f"TrainAcc: {train_acc:.4f}, TrainF1: {macro_f1_t:.4f}, "
            f"ValLoss: {val_loss/val_batches:.4f}, ValAcc@t*: {val_acc_s:.4f}, ValF1@t*: {macro_f1_s:.4f}"
        )

        if writer is not None:
            with writer.as_default():
                tf.summary.scalar("Loss/train", running_loss/max(1,num_batches), step=epoch)
                tf.summary.scalar("Accuracy/train", train_acc, step=epoch)
                tf.summary.scalar("F1/train_macro", macro_f1_t, step=epoch)
                tf.summary.scalar("Loss/val", val_loss/val_batches, step=epoch)
                tf.summary.scalar("Accuracy/val@t*", val_acc_s, step=epoch)
                tf.summary.scalar("F1/val_macro@t*", macro_f1_s, step=epoch)
                tf.summary.scalar("Val/t_star", t_star, step=epoch)
                tf.summary.scalar("Val/FPR0@t*", tinfo["fpr0"], step=epoch)
                tf.summary.scalar("Val/Class0_P", p0_s, step=epoch)
                tf.summary.scalar("Val/Class0_R", r0_s, step=epoch)
                tf.summary.scalar("Val/Class1_P", p1_s, step=epoch)
                tf.summary.scalar("Val/Class1_R", r1_s, step=epoch)
                writer.flush()

        # Best-Model (nach F1@t*)
        if macro_f1_s > best_f1:
            best_f1 = macro_f1_s
            best_info = {
                "epoch": epoch+1,
                "val_macro_f1_t": float(macro_f1_s),
                "val_t_star": float(t_star),
                "val_fpr0_t": float(tinfo["fpr0"]),
                "objective": thresh_objective
            }
            best_model_path = os.path.join(save_path, f"{project}_best_model.pth")
            torch.save({"epoch": epoch+1, "state_dict": model.state_dict(), "t_star": float(t_star)}, best_model_path)
            with open(os.path.join(save_path, f"{project}_best_model_info.json"), "w") as jf:
                json.dump({**best_info, "model_save_path": best_model_path}, jf, indent=2)
            print(f"New best model @ epoch {epoch+1}: macroF1(t*)={macro_f1_s:.4f}, t*={t_star:.4f}")

        if not only_save_best_model:
            ep_path = os.path.join(save_path, f"{project}_model_epoch_{epoch+1}_valF1t_{macro_f1_s:.4f}.pth")
            torch.save(model.state_dict(), ep_path)

        # (4) Hard-Mining für nächste Epoche

        try:
            model.eval()
            train_indices = getattr(dataset_with_cv, "train_idx", list(range(len(dataset_with_cv))))

            # ---------------- Hard NEG ----------------
            neg_indices = [i for i in train_indices if _label_from_pair_idx(i) == 0]
            if len(neg_indices) == 0:
                hard_neg_prev = set()
            else:
                neg_loader = DataLoader(
                    Subset(dataset_with_cv, neg_indices),
                    batch_size=base_bs, shuffle=False, num_workers=0, drop_last=False
                )
                scores, buf = [], []
                offset = 0
                with torch.no_grad():
                    for (x704, xext, labels) in neg_loader:
                        x704 = x704.to(device, non_blocking=True)
                        xext = xext.to(device, non_blocking=True)

                        # <<< hier direkt dein _split_xext
                        x44, x27 = _split_xext(xext, dataset_with_cv)
                        if x44 is not None:
                            x44 = x44.to(device, non_blocking=True)
                        if x27 is not None:
                            x27 = x27.to(device, non_blocking=True)

                        p1 = torch.softmax(model(x704, x44, x27), dim=1)[:, 1].detach().cpu().numpy()

                        n = len(p1)
                        batch_ids = neg_indices[offset:offset + n]
                        offset += n
                        scores.append(p1)
                        buf.extend(batch_ids)

                if len(scores):
                    s = np.concatenate(scores)
                    K = min(int(topk_hard_neg), len(s))
                    topk = np.argpartition(s, -K)[-K:]
                    hard_neg_prev = set(int(buf[i]) for i in topk)
                else:
                    hard_neg_prev = set()

            # ---------------- Hard POS (semi-hard, loss-basiert) ----------------
            warmup_pos_epochs = 10
            mine_every_k = 2
            if (epoch + 1) >= warmup_pos_epochs and ((epoch + 1) % mine_every_k == 0):
                pos_indices = [i for i in train_indices if _label_from_pair_idx(i) == 1]
                if len(pos_indices) == 0:
                    hard_pos_prev = set()
                else:
                    pos_loader = DataLoader(Subset(dataset_with_cv, pos_indices),
                                            batch_size=base_bs, shuffle=False, num_workers=0, drop_last=False)
                    losses, idxbuf = [], []
                    offset = 0
                    with torch.no_grad():
                        for (x704, xext, labels) in pos_loader:
                            x704 = x704.to(device, non_blocking=True)
                            labels = labels.to(device, non_blocking=True).long()
                            x44, x27 = _split_xext(xext, dataset_with_cv)
                            if x44 is not None: x44 = x44.to(device, non_blocking=True)
                            if x27 is not None: x27 = x27.to(device, non_blocking=True)

                            logits = model(x704, x44, x27)
                            p1 = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()  # Tensor auf device

                            ce_each = F.cross_entropy(logits, labels, reduction='none')
                            mask = (p1 >= 0.2) & (p1 <= 0.85)  # Ambigüitätsfenster
                            if mask.any():
                                ce_sel = ce_each[mask].detach().cpu().numpy()
                                bs = labels.size(0)
                                batch_ids = pos_indices[offset:offset + bs]
                                nz = mask.nonzero()
                                sel_idx = nonzero_flat(mask)
                                idxbuf.extend([batch_ids[j] for j in sel_idx])
                                losses.append(ce_sel)
                            offset += labels.size(0)

                    if len(losses):
                        L = np.concatenate(losses)
                        K = min(int(topk_hard_pos), len(L))
                        topk = np.argpartition(L, -K)[-K:]
                        hard_pos_prev = set(int(idxbuf[i]) for i in topk)
                    else:
                        hard_pos_prev = set()
            else:
                if epoch == 0:
                    hard_pos_prev = set()

        except Exception as e:
            print(f"[HardMining] skipped: {e}")
            hard_neg_prev = set()
            hard_pos_prev = set()

        # Cleanup / Monitoring
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            print(f"[Epoch {epoch+1}] VRAM now: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        if process is not None:
            print(f"[Epoch {epoch+1}] RAM: {process.memory_info().rss/(1024**3):.2f} GB")

    # ---------- Test ----------
    print("\nStart Test Loop")
    if best_info is not None:
        t_val_star = float(best_info["val_t_star"])  # nur Info
    else:
        t_val_star = 0.5

    os.makedirs(save_path, exist_ok=True)

    model.eval()
    y_true, y_p1 = [], []

    # fortlaufende Indextabelle für Test-Items (Mapping zurück auf all_pairs)
    test_indices_seq = list(getattr(dataset_with_cv, "test_idx", list(range(len(dataset_with_cv)))))
    ptr = 0
    per_item_indices = []

    with torch.no_grad():
        for (x704, xext, labels) in test_loader:
            x704 = x704.to(device, non_blocking=True)
            xext = xext.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            x44, x27 = _split_xext(xext, dataset_with_cv)
            if x44 is not None: x44 = x44.to(device, non_blocking=True)
            if x27 is not None: x27 = x27.to(device, non_blocking=True)

            logits = model(x704, x44, x27)
            p1 = torch.softmax(logits, dim=1)[:, 1]  # Tensor (device)
            p1_np = p1.detach().cpu().numpy()  # numpy

            # Sammeln
            y_true.extend(labels.cpu().numpy().tolist())
            y_p1.extend(p1_np.tolist())

            # Index-Mapping fortschreiben
            bs = labels.size(0)
            per_item_indices.extend(test_indices_seq[ptr:ptr + bs])
            ptr += bs

    # Zu Arrays
    y_true = np.asarray(y_true, dtype=np.int64)
    p1_all = np.asarray(y_p1, dtype=np.float32)

    # -------- Threshold-Sweep (Score-gestützt) --------
    #cand = np.unique(p1_all)
    #if len(cand) <= 1:
    #    t_grid = np.array([0.5], dtype=np.float32)
    #else:
    #    mid = (cand[1:] + cand[:-1]) / 2.0
    #    t_grid = np.concatenate(([0.0], mid, [1.0])).astype(np.float32)

    # Custom threshold grid: 100 evenly spaced + validation t_val_star
    t_grid = np.linspace(0.0, 1.0, 100, dtype=np.float32)
    if t_val_star not in t_grid:
        t_grid = np.unique(np.append(t_grid, t_val_star)).astype(np.float32)


    def _metrics_fast(y, pred):
        tp = int(np.sum((pred == 1) & (y == 1)))
        tn = int(np.sum((pred == 0) & (y == 0)))
        fp = int(np.sum((pred == 1) & (y == 0)))
        fn = int(np.sum((pred == 0) & (y == 1)))
        # Klasse 1
        prec1 = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec1 = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f11 = (2 * prec1 * rec1) / (prec1 + rec1) if (prec1 + rec1) > 0 else 0.0
        # Klasse 0
        prec0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        rec0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f10 = (2 * prec0 * rec0) / (prec0 + rec0) if (prec0 + rec0) > 0 else 0.0
        macro_f1 = 0.5 * (f10 + f11)
        acc = (tp + tn) / max(1, (tp + tn + fp + fn))
        return {
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "precision_macro": 0.5 * (prec0 + prec1),
            "recall_macro": 0.5 * (rec0 + rec1),
            "f1_macro": macro_f1,
            "class0": {"P": prec0, "R": rec0, "F1": f10},
            "class1": {"P": prec1, "R": rec1, "F1": f11},
            "accuracy": acc,
        }

    results_by_t = {}
    best_key = None  # (f1_macro, -fp01, -t)
    best_t = None
    best_m = None

    for t in t_grid:
        pred = (p1_all >= t).astype(np.int32)
        m = _metrics_fast(y_true, pred)
        fp01 = m["fp"]
        key = (m["f1_macro"], -fp01, -float(t))  # primär macro-F1, dann weniger 0→1, dann höheres t
        if (best_key is None) or (key > best_key):
            best_key = key
            best_t = float(t)
            best_m = m
        results_by_t[f"{t:.6f}"] = {
            "accuracy": m["accuracy"],
            "precision_macro": m["precision_macro"],
            "recall_macro": m["recall_macro"],
            "f1_macro": m["f1_macro"],
            "confusion": {"tn": m["tn"], "fp": m["fp"], "fn": m["fn"], "tp": m["tp"]},
            "class0": m["class0"], "class1": m["class1"]
        }

    # speichern
    sweep_path = os.path.join(save_path, f"{project}_test_threshold_sweep.json")
    with open(sweep_path, "w", encoding="utf-8") as f:
        json.dump({"best_t": best_t, "from_val_t": t_val_star, "results_by_t": results_by_t}, f, indent=2)
    print(f"[TEST] Threshold-Sweep gespeichert: {sweep_path}")
    print(f"Bestes t (TEST, nach macro-F1): t* = {best_t:.6f} "
          f"| macroF1={best_m['f1_macro']:.4f} | Acc={best_m['accuracy']:.4f} | "
          f"P_macro={best_m['precision_macro']:.4f} | R_macro={best_m['recall_macro']:.4f} | "
          f"FP(0→1)={best_m['fp']}")

    # -------- finale Auswertung bei t_best --------
    t_test = best_t
    y_pred = (p1_all >= t_test).astype(np.int64)

    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TP = np.diag(cm)
    FP = cm.sum(axis=0) - TP
    FN = cm.sum(axis=1) - TP
    TN = cm.sum() - (TP + FP + FN)

    prec, rec, f1_cls, supp = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )

    print(f"\nTest @ t*={t_test:.4f}  (val_t*={t_val_star:.4f})")
    print(f"Accuracy: {acc * 100:.2f}%")
    print(f"Precision (macro): {prec_macro:.4f}")
    print(f"Recall (macro):    {rec_macro:.4f}")
    print(f"F1-Score (macro):  {f1_macro:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("Class-wise metrics (TEST):")
    print(f"  Class 0 -> P={prec[0]:.4f} | R={rec[0]:.4f} | F1={f1_cls[0]:.4f} | n={supp[0]}")
    print(f"  Class 1 -> P={prec[1]:.4f} | R={rec[1]:.4f} | F1={f1_cls[1]:.4f} | n={supp[1]}")

    fp_class1 = int(cm[0, 1])
    print(f"Kritische Fehler (0→1): {fp_class1}")

    # ---------- FP-JSON & Aggregationen ----------
    def _parse_key(key):
        p = key.split("_")
        return "_".join(p[:-3]), p[-3], p[-2], int(p[-1])

    def _inc(d, k):
        d[k] = d.get(k, 0) + 1

    def _norm(x):
        return "NA" if x is None else str(x)

    # Sanity: per_item_indices absichern
    if len(per_item_indices) != len(y_true):
        from torch.utils.data import Subset
        ds = getattr(test_loader, "dataset", None)
        if isinstance(ds, Subset):
            candidate = list(map(int, ds.indices))
        elif hasattr(dataset_with_cv, "test_idx"):
            candidate = list(map(int, dataset_with_cv.test_idx))
        else:
            candidate = list(range(len(y_true)))
        per_item_indices = candidate[:len(y_true)]
    assert len(per_item_indices) == len(y_true), "index mapping mismatch"

    fp_list_cls1, fp_list_cls0 = [], []
    by_cls_pair_01, by_inst_pair_01, by_perc_pair_01 = {}, {}, {}
    by_cls_pair_10, by_inst_pair_10, by_perc_pair_10 = {}, {}, {}

    for (yt, yp, p1_val, ds_idx) in zip(y_true, y_pred, p1_all, per_item_indices):
        pair = dataset_with_cv.all_pairs[ds_idx]
        k_ref = pair.get("esf_ref", "")
        k_scan = pair.get("esf_scan", "")
        try:
            cls_r, inst_r, perc_r, idx_r = _parse_key(k_ref)
            cls_s, inst_s, perc_s, idx_s = _parse_key(k_scan)
        except Exception:
            cls_r = inst_r = perc_r = idx_r = cls_s = inst_s = perc_s = idx_s = None

        entry = {
            "ds_index": int(ds_idx),
            "true": int(yt),
            "pred": int(yp),
            "p1": float(p1_val),
            "threshold": float(t_test),
            "esf_ref": k_ref,
            "esf_scan": k_scan,
            "parsed_ref": {"cls": cls_r, "inst": inst_r, "perc": perc_r, "idx": idx_r},
            "parsed_scan": {"cls": cls_s, "inst": inst_s, "perc": perc_s, "idx": idx_s},
        }

        _cls_pair = f"{_norm(cls_r)}|{_norm(cls_s)}"
        _inst_pair = f"{_norm(inst_r)}|{_norm(inst_s)}"
        _perc_pair = f"{_norm(perc_r)}|{_norm(perc_s)}"

        if yt == 0 and yp == 1:
            fp_list_cls1.append(entry)
            _inc(by_cls_pair_01, _cls_pair)
            _inc(by_inst_pair_01, _inst_pair)
            _inc(by_perc_pair_01, _perc_pair)
        elif yt == 1 and yp == 0:
            fp_list_cls0.append(entry)
            _inc(by_cls_pair_10, _cls_pair)
            _inc(by_inst_pair_10, _inst_pair)
            _inc(by_perc_pair_10, _perc_pair)

    def _sorted_items(d):
        return [{"pair": k, "count": v} for k, v in sorted(d.items(), key=lambda kv: kv[1], reverse=True)]

    fp_json = {
        "project": project,
        "threshold": float(t_test),
        "counts": {
            "fp_class1_0to1": len(fp_list_cls1),
            "fp_class0_1to0": len(fp_list_cls0)
        },
        "aggregation_by_perc": {
            "class1_0to1": {k: v for k, v in by_perc_pair_01.items()},
            "class0_1to0": {k: v for k, v in by_perc_pair_10.items()}
        },
        "aggregation_by_pairs": {
            "0to1": {
                "by_cls_pair_hist": by_cls_pair_01,
                "by_inst_pair_hist": by_inst_pair_01,
                "by_perc_pair_hist": by_perc_pair_01,
                "top_by_cls_pair": _sorted_items(by_cls_pair_01),
                "top_by_inst_pair": _sorted_items(by_inst_pair_01),
                "top_by_perc_pair": _sorted_items(by_perc_pair_01),
            },
            "1to0": {
                "by_cls_pair_hist": by_cls_pair_10,
                "by_inst_pair_hist": by_inst_pair_10,
                "by_perc_pair_hist": by_perc_pair_10,
                "top_by_cls_pair": _sorted_items(by_cls_pair_10),
                "top_by_inst_pair": _sorted_items(by_inst_pair_10),
                "top_by_perc_pair": _sorted_items(by_perc_pair_10),
            }
        },
        "fp_details": {
            "class1_0to1": fp_list_cls1,
            "class0_1to0": fp_list_cls0
        }
    }

    # --- ROBUST SAVE (mit Fehlerausgabe) ---
    fp_json_path = os.path.join(save_path, f"{project}_test_false_positives.json")
    try:
        with open(fp_json_path, "w", encoding="utf-8") as jf:
            json.dump(fp_json, jf, indent=2, ensure_ascii=False)
        print(f"Saved FP details to {fp_json_path}")
    except Exception as e:
        print(f"[SAVE-ERROR] FP JSON: {e}")

    # Excel-Export
    conf_df = pd.DataFrame(cm, index=["True_0", "True_1"], columns=["Pred_0", "Pred_1"])
    per_class = pd.DataFrame({
        "Class": [0, 1], "Support": supp, "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "Precision": prec, "Recall": rec, "F1": f1_cls
    })
    summary = pd.DataFrame({
        "Metric": ["Accuracy", "Precision(macro)", "Recall(macro)", "F1(macro)", "FP_class1(0→1)", "t_star_test",
                   "t_star_val"],
        "Value": [acc, prec_macro, rec_macro, f1_macro, fp_class1, t_test, t_val_star]
    })
    xlsx_path = os.path.join(save_path, f"{project}_test_metrics.xlsx")
    try:
        with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as xw:
            per_class.to_excel(xw, sheet_name="Per_Class", index=False)
            conf_df.to_excel(xw, sheet_name="Confusion_Matrix")
            summary.to_excel(xw, sheet_name="Summary", index=False)
        print(f"Saved test metrics to {xlsx_path}")
    except Exception as e:
        print(f"[SAVE-ERROR] Excel: {e}")

    total_time = time.time() - total_start
    tt_path = os.path.join(save_path, f"{project}_total_training_time.txt")
    try:
        with open(tt_path, "w") as f:
            f.write(f"Total training time: {total_time:.2f} seconds\n")
        print(f"Saved total training time to {tt_path}")
    except Exception as e:
        print(f"[SAVE-ERROR] total_training_time: {e}")
    print(f"Total training time: {total_time:.2f} s")

    return acc, prec_macro, rec_macro, f1_macro




#####
def train_and_evaluate_cnn_xfeat_thresh_neg(
    model,
    dataset_with_cv,
    train_loader,      # initialer Loader (nur für steps_per_epoch von OneCycle)
    val_loader,
    test_loader,
    epochs=150,
    learning_rate=3e-4,
    device="cpu",
    save_path="./models",
    project="default",
    only_save_best_model=True,
    cv_info=None,
    fold=None,
    # ---- Threshold-Optionen
    fp0_max_rate=0.006,             # max erlaubte 0->1-Fehlrate (FPR0) auf VAL
    thresh_grid=None,               # z.B. np.linspace(0.5, 0.99, 100)
    thresh_objective="macro_f1",    # "macro_f1" | "f1_class1" | "recall_class1"
    # ---- Hard-Mining
    max_negatives_per_epoch=500_000,
    boost_hard_neg=8.0,
    topk_hard_neg=100_000,
    # >>> NEU: Hard-Positives
    boost_hard_pos=1.0, #3
    topk_hard_pos=30_000,#5_000
    # ---- Hard-Mining (NEU: toggles & boosts)
    use_sc_hard_negs=False,  # same-class in Label 0 boosten
    boost_sc_hard_neg=6.0,  # Boost-Faktor dafür
    sc_hard_neg_max=None,  # ggf. kappen (z.B. 50_000)

    use_perc_hard_pos=False,  # positive Buckets boosten
    perc_hard_pos_values=(35, 40, 45, 50, 55),
    perc_side="scan",  # "scan" | "ref" | "either" | "both"
    boost_perc_hard_pos=3.0,
    # ---- Class-Weights
    class_weights_vec=[1.0, 3.0],         # z.B. [4.0, 1.0]; None => auto versuchen [4.0,1.0] war für 87% f1 mit negative hard minind.   1,3 für positive
    weight_decay=1e-5,
    grad_accum_steps=4, balanced = True
):
    """
    Trainiert mit Hard-Negatives & Hard-Positives und wählt pro Epoche einen Threshold t*
    auf der Validation (FPR0 <= fp0_max_rate, Ziel=thresh_objective).
    Speichert Best-Model (nach F1@t*) und exportiert Testmetriken inkl. per-Klasse.
    """
    import os, time, json, gc
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
    from sklearn.metrics import (
        confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
        precision_recall_fscore_support
    )

    # optional: RAM-Monitor
    try:
        import psutil
        process = psutil.Process(os.getpid())
    except Exception:
        process = None

    # optional: TensorBoard (nur CPU-Benutzung, kein GPU-Claim)
    try:
        import tensorflow as tf
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass
    except Exception:
        tf = None

    os.makedirs(save_path, exist_ok=True)
    writer = tf.summary.create_file_writer(os.path.join(save_path, "logs")) if tf is not None else None

    # ---------- Hilfsfunktionen ----------
    def _get_y_mapped_from_pair(p):
        # nimmt dein Mapping, falls vorhanden; sonst Fallback
        try:
            return globals()["_y_mapped_from_pair"](p)
        except Exception:
            y = p.get("label", None)
            if y is None or y == 0:
                return None
            return 0 if y == 2 else 1

    def _label_from_pair_idx(i):
        ym = _get_y_mapped_from_pair(dataset_with_cv.all_pairs[i])
        return None if ym is None else int(ym)

    def metrics_from_counts(tn, fp, fn, tp):
        # per Klasse und macro
        prec1 = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec1  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f11   = (2*prec1*rec1)/(prec1+rec1) if (prec1+rec1) > 0 else 0.0

        prec0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        rec0  = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f10   = (2*prec0*rec0)/(prec0+rec0) if (prec0+rec0) > 0 else 0.0

        macro_f1 = 0.5 * (f10 + f11)
        return (prec0, rec0, f10), (prec1, rec1, f11), macro_f1

    def pick_threshold(p1, y, grid, max_fp0_rate, objective="macro_f1"):
        # gibt best_t, dict(metrics) zurück
        if grid is None or len(grid) == 0:
            grid = np.linspace(0.5, 0.99, 100)

        best = None
        best_key = None

        for t in grid:
            pred = (p1 >= t).astype(np.int32)
            cm = confusion_matrix(y, pred, labels=[0, 1])
            if cm.shape != (2, 2):
                tn = fp = fn = tp = 0
                if cm.shape == (1, 1):  # nur eine Klasse gesehen
                    tn = int(cm[0, 0]) if (y == 0).all() else 0
                    tp = int(cm[0, 0]) if (y == 1).all() else 0
            else:
                tn, fp, fn, tp = int(cm[0,0]), int(cm[0,1]), int(cm[1,0]), int(cm[1,1])

            denom = tn + fp
            fpr0 = (fp / denom) if denom > 0 else 0.0

            (p0,r0,f10),(p1c,r1c,f11),macro_f1 = metrics_from_counts(tn, fp, fn, tp)

            if objective == "f1_class1":
                key_primary = f11
            elif objective == "recall_class1":
                key_primary = r1c
            else:
                key_primary = macro_f1

            feasible = (fpr0 <= max_fp0_rate)
            key = (1 if feasible else 0, key_primary, -fpr0, -t)

            cand = {
                "t": float(t),
                "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                "fpr0": float(fpr0),
                "macro_f1": float(macro_f1),
                "p0": float(p0), "r0": float(r0), "f10": float(f10),
                "p1": float(p1c), "r1": float(r1c), "f11": float(f11),
                "feasible": bool(feasible)
            }
            if (best_key is None) or (key > best_key):
                best = cand
                best_key = key

        return best["t"], best

    import torch.nn.functional as F  # <— fehlte für CE

    def _parse_esf_key(key: str):
        # Erwartet Format "..._<inst>_<perc>_<idx>"
        try:
            parts = key.split("_")
            cls = "_".join(parts[:-3]) if len(parts) >= 4 else None
            inst = parts[-3] if len(parts) >= 3 else None
            perc = int(parts[-2]) if len(parts) >= 2 and parts[-2].isdigit() else None
            idx  = int(parts[-1]) if len(parts) >= 1 and parts[-1].isdigit() else None
            return cls, inst, perc, idx
        except Exception:
            return None, None, None, None

    def _pair_meta_from_idx(i):
        p = dataset_with_cv.all_pairs[i]
        k_ref = p.get("esf_ref", "")
        k_scan = p.get("esf_scan", "")
        cls_r, inst_r, perc_r, _ = _parse_esf_key(k_ref)
        cls_s, inst_s, perc_s, _ = _parse_esf_key(k_scan)
        return (cls_r, inst_r, perc_r), (cls_s, inst_s, perc_s)

    def _same_class(i):
        (cr, _, _), (cs, _, _) = _pair_meta_from_idx(i)
        return (cr is not None) and (cs is not None) and (cr == cs)

    def _perc_hit(i, values, side="scan"):
        (cr, ir, pr), (cs, is_, ps) = _pair_meta_from_idx(i)
        V = set(int(v) for v in values)
        if side == "scan":
            return (ps is not None) and (ps in V)
        elif side == "ref":
            return (pr is not None) and (pr in V)
        elif side == "either":
            return ((ps is not None) and (ps in V)) or ((pr is not None) and (pr in V))
        elif side == "both":
            return ((ps is not None) and (ps in V)) and ((pr is not None) and (pr in V))
        else:
            return False

    # >>> Sampler erweitert: Hard-Neg UND Hard-Pos

    def make_epoch_sampler_balanced(
            train_idx, all_pairs,
            neg_multiple=2.0,  # z.B. 1.5–2.5 (pos:neg ~ 1:1.5..1:2.5)
            keep_all_positives=True,
            hard_neg=None, boost_neg=3.0,
            hard_pos=None, boost_pos=8.0
    ):
        import numpy as np, torch
        from torch.utils.data import WeightedRandomSampler

        hard_neg = set(hard_neg or [])
        hard_pos = set(hard_pos or [])

        pos_idx, neg_idx = [], []
        for i in train_idx:
            y = _get_y_mapped_from_pair(all_pairs[i])
            if y is None:
                continue
            (pos_idx if y == 1 else neg_idx).append(i)

        # Negatives proportional zu Positives samplen
        target_negs = int(len(pos_idx) * float(neg_multiple))
        if len(neg_idx) > target_negs:
            neg_sampled = np.random.choice(neg_idx, size=target_negs, replace=False)
        else:
            neg_sampled = np.asarray(neg_idx, dtype=np.int64)

        indices = np.asarray(pos_idx, dtype=np.int64) if keep_all_positives else np.array([], dtype=np.int64)
        if len(neg_sampled) > 0:
            indices = np.concatenate([indices, neg_sampled])

        # Gewichte: HardPos stark boosten, HardNeg moderat
        weights = np.ones(len(indices), dtype=np.float32)
        idx_set = set(indices.tolist())
        for k, idx in enumerate(indices):
            if idx in hard_pos:
                weights[k] = float(boost_pos)
            elif idx in hard_neg:
                weights[k] = float(boost_neg)

        sampler = WeightedRandomSampler(
            weights=torch.tensor(weights, dtype=torch.float32),
            num_samples=len(indices),
            replacement=True
        )
        return sampler, indices

    def make_epoch_sampler(
            train_idx, all_pairs, max_negatives,
            keep_all_positives=True,
            hard_neg=None, boost_neg=5.0,
            hard_pos=None, boost_pos=3.0,
            # NEU:
            hard_neg_sc=None, boost_neg_sc=6.0,
            hard_pos_perc=None, boost_pos_perc=3.0,
    ):
        hard_neg = set(hard_neg or [])
        hard_pos = set(hard_pos or [])
        hard_neg_sc = set(hard_neg_sc or [])
        hard_pos_perc = set(hard_pos_perc or [])

        pos_idx, neg_idx = [], []
        for i in train_idx:
            ym = _get_y_mapped_from_pair(all_pairs[i])
            if ym is None:
                continue
            (pos_idx if ym == 1 else neg_idx).append(i)

        if (max_negatives is not None) and (len(neg_idx) > max_negatives):
            neg_sampled = np.random.choice(neg_idx, size=max_negatives, replace=False)
        else:
            neg_sampled = np.asarray(neg_idx, dtype=np.int64)

        indices = np.asarray(pos_idx, dtype=np.int64) if keep_all_positives else np.array([], dtype=np.int64)
        if len(neg_sampled) > 0:
            indices = np.concatenate([indices, neg_sampled])

        weights = np.ones(len(indices), dtype=np.float32)
        for k, idx in enumerate(indices):
            if idx in hard_neg:       weights[k] *= float(boost_neg)
            if idx in hard_pos:       weights[k] *= float(boost_pos)
            if idx in hard_neg_sc:    weights[k] *= float(boost_neg_sc)
            if idx in hard_pos_perc:  weights[k] *= float(boost_pos_perc)

        sampler = WeightedRandomSampler(
            weights=torch.tensor(weights, dtype=torch.float32),
            num_samples=len(indices),
            replacement=True
        )
        return sampler, indices

    # ---------- Setup ----------
    device = torch.device(device)
    model = model.to(device)

    # Class-Weights
    if class_weights_vec is not None:
        class_weights = torch.tensor(class_weights_vec, dtype=torch.float32, device=device)
    else:
        try:
            class_weights = make_ce_weights_from_dataset(dataset_with_cv, alpha_neg=1.6, device=device)
        except Exception:
            class_weights = torch.tensor([1.0, 1.0], dtype=torch.float32, device=device)
    print("[CE weights]", class_weights.tolist())

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    steps_per_epoch = max(1, len(train_loader))
    onecycle = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=max(learning_rate, 1e-3), epochs=epochs,
        steps_per_epoch=steps_per_epoch, div_factor=25.0, final_div_factor=1e4
    )
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, threshold=1e-4,
        cooldown=1, min_lr=1e-6
    )
    use_plateau_after = max(4, epochs - 4)

    # History
    best_f1 = -1.0
    best_info = None

    hard_neg_prev = set()
    hard_pos_prev = set()   # <<< NEU
    base_bs = getattr(train_loader, "batch_size", 64)

    # ---------- Training ----------
    total_start = time.time()
    
    # ---- Static Sets für Flags (kein Model-Forward nötig)
    train_indices_all = getattr(dataset_with_cv, "train_idx", list(range(len(dataset_with_cv))))

    static_sc_hard_neg = set()
    if use_sc_hard_negs:
        for i in train_indices_all:
            y = _label_from_pair_idx(i)
            if y == 0 and _same_class(i):
                static_sc_hard_neg.add(i)
        if sc_hard_neg_max is not None and len(static_sc_hard_neg) > sc_hard_neg_max:
            static_sc_hard_neg = set(np.random.choice(list(static_sc_hard_neg), size=int(sc_hard_neg_max), replace=False))

    static_perc_hard_pos = set()
    if use_perc_hard_pos:
        for i in train_indices_all:
            y = _label_from_pair_idx(i)
            if y == 1 and _perc_hit(i, perc_hard_pos_values, side=perc_side):
                static_perc_hard_pos.add(i)

    print(f"[StaticHard] same-class neg: {len(static_sc_hard_neg)} | "
          f"perc-hard pos: {len(static_perc_hard_pos)} ({perc_side} in {list(perc_hard_pos_values)})")
    
    for epoch in range(epochs):
        # (1) Epoche: Sampler bauen (beide Hard-Sets/Boosts berücksichtigen)
        try:
            if balance:
                sampler, ep_indices = make_epoch_sampler_balanced(
                    dataset_with_cv.train_idx,
                    dataset_with_cv.all_pairs,
                    neg_multiple=2.0,  # <- gerne 1.5–2.5 testen
                    keep_all_positives=True,
                    hard_neg=hard_neg_prev, boost_neg=3.0,
                    hard_pos=hard_pos_prev, boost_pos=8.0
                )
                tr_loader = DataLoader(
                    Subset(dataset_with_cv, ep_indices),
                    batch_size=base_bs, sampler=sampler, num_workers=0, drop_last=False
                )
            else:

                sampler, ep_indices = make_epoch_sampler(
                    dataset_with_cv.train_idx,
                    dataset_with_cv.all_pairs,
                    max_negatives=max_negatives_per_epoch,
                    keep_all_positives=True,
                    hard_neg=hard_neg_prev, boost_neg=boost_hard_neg,
                    hard_pos=hard_pos_prev, boost_pos=boost_hard_pos,
                    # NEU: statische Sets
                    hard_neg_sc=static_sc_hard_neg, boost_neg_sc=boost_sc_hard_neg,
                    hard_pos_perc=static_perc_hard_pos, boost_pos_perc=boost_perc_hard_pos
                )

            tr_loader = DataLoader(
                Subset(dataset_with_cv, ep_indices),
                batch_size=base_bs, sampler=sampler, num_workers=0, drop_last=False
            )

            print(f"[HardMining] next epoch: "
                  f"dyn_hard_neg(topk)={len(hard_neg_prev)} (×{boost_hard_neg}), "
                  f"dyn_hard_pos(loss)={len(hard_pos_prev)} (×{boost_hard_pos}), "
                  f"static_sameclass_neg={len(static_sc_hard_neg)} (×{boost_sc_hard_neg}), "
                  f"static_perc_pos={len(static_perc_hard_pos)} (×{boost_perc_hard_pos})")
        except Exception as e:
            print(f"[Sampler-Fallback] {e}")
            tr_loader = train_loader  # initialer

        # (2) Train
        model.train()
        running_loss = 0.0
        tn_tr = fp_tr = fn_tr = tp_tr = 0
        optimizer.zero_grad(set_to_none=True)
        accum = 0
        num_batches = len(tr_loader)

        for b, (x704, xext, labels) in enumerate(tr_loader, start=1):
            x704   = x704.to(device, non_blocking=True)
            xext   = xext.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(x704, xext)
            loss = criterion(logits, labels)

            loss = loss / max(1, grad_accum_steps)
            loss.backward()
            accum += 1

            if (accum == grad_accum_steps) or (b == num_batches):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                onecycle.step()
                accum = 0

            running_loss += loss.item() * max(1, grad_accum_steps)

            with torch.no_grad():
                pred = logits.argmax(1)
                cm = confusion_matrix(labels.detach().cpu().numpy(),
                                      pred.detach().cpu().numpy(), labels=[0, 1])
                if cm.shape == (2,2):
                    tn_tr += int(cm[0,0]); fp_tr += int(cm[0,1])
                    fn_tr += int(cm[1,0]); tp_tr += int(cm[1,1])

            del logits, loss, pred
            if torch.cuda.is_available() and (b % 100 == 0):
                torch.cuda.empty_cache()

        train_total = tn_tr + fp_tr + fn_tr + tp_tr
        train_acc = (tn_tr + tp_tr) / max(1, train_total)
        (p0_t, r0_t, f10_t), (p1_t, r1_t, f11_t), macro_f1_t = metrics_from_counts(tn_tr, fp_tr, fn_tr, tp_tr)

        # (3) Validation – sammle p1 und y für Threshold-Suche
        model.eval()
        val_loss = 0.0
        tn_v = fp_v = fn_v = tp_v = 0
        val_p1_all, y_val_all = [], []

        with torch.no_grad():
            for (x704, xext, labels) in val_loader:
                x704   = x704.to(device, non_blocking=True)
                xext   = xext.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                logits = model(x704, xext)
                loss = criterion(logits, labels)
                val_loss += loss.item()

                probs = torch.softmax(logits, dim=1)
                p1 = probs[:, 1].detach().cpu().numpy()
                val_p1_all.append(p1)
                y_val_all.append(labels.detach().cpu().numpy())

                pred_max = logits.argmax(1)
                cm = confusion_matrix(labels.cpu().numpy(), pred_max.cpu().numpy(), labels=[0, 1])
                if cm.shape == (2,2):
                    tn_v += int(cm[0,0]); fp_v += int(cm[0,1])
                    fn_v += int(cm[1,0]); tp_v += int(cm[1,1])

                del logits, loss, probs, pred_max

        p1_val = np.concatenate(val_p1_all) if len(val_p1_all) else np.array([], dtype=np.float32)
        y_val  = np.concatenate(y_val_all)  if len(y_val_all)  else np.array([], dtype=np.int64)

        # Threshold-Suche
        if thresh_grid is None:
            grid = np.linspace(0.5, 0.99, 100)
        else:
            grid = np.asarray(thresh_grid, dtype=np.float32)

        t_star, tinfo = pick_threshold(
            p1=p1_val, y=y_val, grid=grid,
            max_fp0_rate=fp0_max_rate, objective=thresh_objective
        )

        # Val-Metrik @ t*
        pred_t = (p1_val >= t_star).astype(np.int32)
        cm_t = confusion_matrix(y_val, pred_t, labels=[0, 1])
        if cm_t.shape == (2,2):
            tn_s, fp_s, fn_s, tp_s = int(cm_t[0,0]), int(cm_t[0,1]), int(cm_t[1,0]), int(cm_t[1,1])
        else:
            tn_s = fp_s = fn_s = tp_s = 0
        (p0_s, r0_s, f10_s), (p1_s, r1_s, f11_s), macro_f1_s = metrics_from_counts(tn_s, fp_s, fn_s, tp_s)
        val_acc_s = (tn_s + tp_s) / max(1, (tn_s + fp_s + fn_s + tp_s))
        fp01_val = fp_s

        print("\n[VAL @ argmax] (nur Info)")
        (p0_a, r0_a, f10_a), (p1_a, r1_a, f11_a), macro_f1_a = metrics_from_counts(tn_v, fp_v, fn_v, tp_v)
        print(f"  macroF1(argmax)={macro_f1_a:.4f}")

        print(f"[VAL Threshold-Search] fp0_max_rate={fp0_max_rate:.4f} | best t*={t_star:.4f} "
              f"| feasible={tinfo['feasible']} | fpr0={tinfo['fpr0']:.5f}")
        print(f"  Class 0 -> P={p0_s:.4f} | R={r0_s:.4f} | F1={f10_s:.4f}")
        print(f"  Class 1 -> P={p1_s:.4f} | R={r1_s:.4f} | F1={f11_s:.4f}")
        print(f"  macroF1(t*)={macro_f1_s:.4f} | crit.errors 0→1 = {fp01_val}")

        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1}/{epochs} | lr={cur_lr:.6g}")

        if epoch + 1 >= use_plateau_after:
            plateau.step(macro_f1_s)

        with open(os.path.join(save_path, f"{project}_train_loss.txt"), "a") as f:
            f.write(f"{epoch+1}; {running_loss/max(1,num_batches):.4f}\n")
        with open(os.path.join(save_path, f"{project}_train_accuracy.txt"), "a") as f:
            f.write(f"{train_acc:.4f}\n")
        with open(os.path.join(save_path, f"{project}_train_f1.txt"), "a") as f:
            f.write(f"{macro_f1_t:.4f}\n")

        val_batches = max(1, len(val_loader))
        with open(os.path.join(save_path, f"{project}_val_loss.txt"), "a") as f:
            f.write(f"{epoch+1}; {val_loss/val_batches:.4f}\n")
        with open(os.path.join(save_path, f"{project}_val_accuracy.txt"), "a") as f:
            f.write(f"{val_acc_s:.4f}\n")
        with open(os.path.join(save_path, f"{project}_val_f1.txt"), "a") as f:
            f.write(f"{macro_f1_s:.4f}\n")

        print(
            f"Epoch {epoch+1}/{epochs}, "
            f"TrainLoss: {running_loss/max(1,num_batches):.4f}, "
            f"TrainAcc: {train_acc:.4f}, TrainF1: {macro_f1_t:.4f}, "
            f"ValLoss: {val_loss/val_batches:.4f}, ValAcc@t*: {val_acc_s:.4f}, ValF1@t*: {macro_f1_s:.4f}"
        )

        if writer is not None:
            with writer.as_default():
                tf.summary.scalar("Loss/train", running_loss/max(1,num_batches), step=epoch)
                tf.summary.scalar("Accuracy/train", train_acc, step=epoch)
                tf.summary.scalar("F1/train_macro", macro_f1_t, step=epoch)
                tf.summary.scalar("Loss/val", val_loss/val_batches, step=epoch)
                tf.summary.scalar("Accuracy/val@t*", val_acc_s, step=epoch)
                tf.summary.scalar("F1/val_macro@t*", macro_f1_s, step=epoch)
                tf.summary.scalar("Val/t_star", t_star, step=epoch)
                tf.summary.scalar("Val/FPR0@t*", tinfo["fpr0"], step=epoch)
                tf.summary.scalar("Val/Class0_P", p0_s, step=epoch)
                tf.summary.scalar("Val/Class0_R", r0_s, step=epoch)
                tf.summary.scalar("Val/Class1_P", p1_s, step=epoch)
                tf.summary.scalar("Val/Class1_R", r1_s, step=epoch)
                writer.flush()

        # Best-Model (nach F1@t*)
        if macro_f1_s > best_f1:
            best_f1 = macro_f1_s
            best_info = {
                "epoch": epoch+1,
                "val_macro_f1_t": float(macro_f1_s),
                "val_t_star": float(t_star),
                "val_fpr0_t": float(tinfo["fpr0"]),
                "objective": thresh_objective
            }
            best_model_path = os.path.join(save_path, f"{project}_best_model.pth")
            torch.save({"epoch": epoch+1, "state_dict": model.state_dict(), "t_star": float(t_star)}, best_model_path)
            with open(os.path.join(save_path, f"{project}_best_model_info.json"), "w") as jf:
                json.dump({**best_info, "model_save_path": best_model_path}, jf, indent=2)
            print(f"New best model @ epoch {epoch+1}: macroF1(t*)={macro_f1_s:.4f}, t*={t_star:.4f}")

        if not only_save_best_model:
            ep_path = os.path.join(save_path, f"{project}_model_epoch_{epoch+1}_valF1t_{macro_f1_s:.4f}.pth")
            torch.save(model.state_dict(), ep_path)

        # (4) Hard-Mining für nächste Epoche
        # (4) Hard-Mining für nächste Epoche
        try:
            model.eval()
            train_indices = getattr(dataset_with_cv, "train_idx", list(range(len(dataset_with_cv))))

            # ---------------- Hard NEG ----------------
            neg_indices = [i for i in train_indices if _label_from_pair_idx(i) == 0]
            if len(neg_indices) == 0:
                hard_neg_prev = set()
            else:
                neg_loader = DataLoader(Subset(dataset_with_cv, neg_indices),
                                        batch_size=base_bs, shuffle=False, num_workers=0, drop_last=False)
                scores, buf = [], []
                offset = 0
                with torch.no_grad():
                    for (x704, xext, labels) in neg_loader:
                        x704 = x704.to(device, non_blocking=True)
                        x44, x27 = _split_xext(xext, dataset_with_cv)
                        if x44 is not None: x44 = x44.to(device, non_blocking=True)
                        if x27 is not None: x27 = x27.to(device, non_blocking=True)

                        logits = model(x704, x44, x27)
                        p1 = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()  # << wichtig

                        n = len(p1)
                        batch_ids = neg_indices[offset:offset + n]
                        offset += n
                        scores.append(p1);
                        buf.extend(batch_ids)
                if len(scores):
                    s = np.concatenate(scores)
                    K = min(int(topk_hard_neg), len(s))
                    topk = np.argpartition(s, -K)[-K:]
                    hard_neg_prev = set(int(buf[i]) for i in topk)
                else:
                    hard_neg_prev = set()

            # ---------------- Hard POS (semi-hard, loss-basiert) ----------------
            # optional: Warmup + Kadenz
            warmup_pos_epochs = 10
            mine_every_k = 2
            if (epoch + 1) >= warmup_pos_epochs and ((epoch + 1) % mine_every_k == 0):
                pos_indices = [i for i in train_indices if _label_from_pair_idx(i) == 1]
                if len(pos_indices) == 0:
                    hard_pos_prev = set()
                else:
                    pos_loader = DataLoader(Subset(dataset_with_cv, pos_indices),
                                            batch_size=base_bs, shuffle=False, num_workers=0, drop_last=False)
                    losses, idxbuf = [], []
                    offset = 0
                    with torch.no_grad():
                        for (x704, xext, labels) in pos_loader:
                            x704 = x704.to(device, non_blocking=True)
                            xext = xext.to(device, non_blocking=True)
                            # WICHTIG: Labels auch auf dasselbe Device + richtiger dtype
                            labels = labels.to(device, non_blocking=True).long()

                            logits = model(x704, xext)  # logits auf device
                            p1 = torch.softmax(logits, dim=1)[:, 1]

                            # Cross-Entropy pro Sample — Targets müssen auf GLEICHEM Device wie logits sein
                            ce_each = F.cross_entropy(logits, labels, reduction='none')

                            # Ambiguitätsfenster
                            mask = (p1 >= 0.25) & (p1 <= 0.80)
                            if mask.any():
                                ce_sel = ce_each[mask].detach().cpu().numpy()
                                bs = labels.size(0)
                                batch_ids = pos_indices[offset:offset + bs]
                                sel_idx = mask.nonzero(as_tuple=True)[0].cpu().numpy()
                                idxbuf.extend([batch_ids[j] for j in sel_idx])
                                losses.append(ce_sel)
                            offset += labels.size(0)

                    if len(losses):
                        L = np.concatenate(losses)
                        K = min(int(topk_hard_pos), len(L))
                        topk = np.argpartition(L, -K)[-K:]
                        hard_pos_prev = set(int(idxbuf[i]) for i in topk)
                    else:
                        hard_pos_prev = set()
            else:
                # solange Warmup/Kadenz nicht greift: nichts boosten
                if epoch == 0:
                    hard_pos_prev = set()

            #print(f"[HardMining] next epoch: hard_neg={len(hard_neg_prev)} (×{boost_hard_neg}), "
            #      f"hard_pos={len(hard_pos_prev)} (×{boost_hard_pos})")

        except Exception as e:
            print(f"[HardMining] skipped: {e}")
            hard_neg_prev = set()
            hard_pos_prev = set()

        # Cleanup / Monitoring
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            print(f"[Epoch {epoch+1}] VRAM now: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        if process is not None:
            print(f"[Epoch {epoch+1}] RAM: {process.memory_info().rss/(1024**3):.2f} GB")

    # ---------- Test ----------
    print("\nStart Test Loop")
    if best_info is not None:
        t_test = float(best_info["val_t_star"])
    else:
        t_test = 0.5

    # Sicherstellen, dass der Pfad existiert
    os.makedirs(save_path, exist_ok=True)

    model.eval()
    y_true, y_pred, y_p1 = [], [], []

    # >>> NEU: baue eine fortlaufende Indextabelle für Test-Items
    # Falls dein Dataset beim Laden in test_loader die Reihenfolge von dataset_with_cv.test_idx nutzt:
    test_indices_seq = list(getattr(dataset_with_cv, "test_idx", list(range(len(dataset_with_cv)))))
    ptr = 0
    per_item_indices = []

    with torch.no_grad():
        for (x704, xext, labels) in test_loader:
            x704 = x704.to(device, non_blocking=True)
            xext = xext.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            p1 = torch.softmax(model(x704, xext), dim=1)[:, 1]
            pred = (p1 >= t_test).long()

            # >>> NEU: die passenden globalen Datensatz-Indizes für diese Batch übernehmen
            bs = labels.size(0)
            per_item_indices.extend(test_indices_seq[ptr:ptr + bs])
            ptr += bs

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(pred.cpu().numpy().tolist())
            y_p1.extend(p1.cpu().numpy().tolist())

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TP = np.diag(cm)
    FP = cm.sum(axis=0) - TP
    FN = cm.sum(axis=1) - TP
    TN = cm.sum() - (TP + FP + FN)

    prec, rec, f1_cls, supp = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )

    print(f"Test @ t*={t_test:.4f}")
    print(f"Accuracy: {acc * 100:.2f}%")
    print(f"Precision (macro): {prec_macro:.4f}")
    print(f"Recall (macro):    {rec_macro:.4f}")
    print(f"F1-Score (macro):  {f1_macro:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("Class-wise metrics (TEST):")
    print(f"  Class 0 -> P={prec[0]:.4f} | R={rec[0]:.4f} | F1={f1_cls[0]:.4f} | n={supp[0]}")
    print(f"  Class 1 -> P={prec[1]:.4f} | R={rec[1]:.4f} | F1={f1_cls[1]:.4f} | n={supp[1]}")

    fp_class1 = int(cm[0, 1])
    print(f"Kritische Fehler (0→1): {fp_class1}")

    # ---------- FP-JSON & Aggregationen ----------
    def _parse_key(key):
        p = key.split("_")
        return "_".join(p[:-3]), p[-3], p[-2], int(p[-1])

    def _inc(d, k):
        d[k] = d.get(k, 0) + 1

    def _norm(x):
        return "NA" if x is None else str(x)

    fp_list_cls1, fp_list_cls0 = [], []
    by_cls_pair_01, by_inst_pair_01, by_perc_pair_01 = {}, {}, {}
    by_cls_pair_10, by_inst_pair_10, by_perc_pair_10 = {}, {}, {}

    for (yt, yp, p1_val, ds_idx) in zip(y_true, y_pred, y_p1, per_item_indices):
        pair = dataset_with_cv.all_pairs[ds_idx]
        k_ref = pair.get("esf_ref", "")
        k_scan = pair.get("esf_scan", "")
        try:
            cls_r, inst_r, perc_r, idx_r = _parse_key(k_ref)
            cls_s, inst_s, perc_s, idx_s = _parse_key(k_scan)
        except Exception:
            cls_r = inst_r = perc_r = idx_r = cls_s = inst_s = perc_s = idx_s = None

        entry = {
            "ds_index": int(ds_idx),
            "true": int(yt),
            "pred": int(yp),
            "p1": float(p1_val),
            "threshold": float(t_test),
            "esf_ref": k_ref,
            "esf_scan": k_scan,
            "parsed_ref": {"cls": cls_r, "inst": inst_r, "perc": perc_r, "idx": idx_r},
            "parsed_scan": {"cls": cls_s, "inst": inst_s, "perc": perc_s, "idx": idx_s},
        }

        _cls_pair = f"{_norm(cls_r)}|{_norm(cls_s)}"
        _inst_pair = f"{_norm(inst_r)}|{_norm(inst_s)}"
        _perc_pair = f"{_norm(perc_r)}|{_norm(perc_s)}"

        if yt == 0 and yp == 1:
            fp_list_cls1.append(entry)
            _inc(by_cls_pair_01, _cls_pair)
            _inc(by_inst_pair_01, _inst_pair)
            _inc(by_perc_pair_01, _perc_pair)
        elif yt == 1 and yp == 0:
            fp_list_cls0.append(entry)
            _inc(by_cls_pair_10, _cls_pair)
            _inc(by_inst_pair_10, _inst_pair)
            _inc(by_perc_pair_10, _perc_pair)

    def _sorted_items(d):
        return [{"pair": k, "count": v} for k, v in sorted(d.items(), key=lambda kv: kv[1], reverse=True)]

    # Sanity: sicherstellen, dass per_item_indices existiert und Längen passen
    if 'per_item_indices' not in locals():
        per_item_indices = []

    if len(per_item_indices) != len(y_true):
        # Fallback: Indizes aus dem Test-Loader rekonstruieren
        from torch.utils.data import Subset
        ds = getattr(test_loader, "dataset", None)
        if isinstance(ds, Subset):
            candidate = list(map(int, ds.indices))
        elif hasattr(dataset_with_cv, "test_idx"):
            candidate = list(map(int, dataset_with_cv.test_idx))
        else:
            candidate = list(range(len(y_true)))
        per_item_indices = candidate[:len(y_true)]

    print(f"[DEBUG] len(y_true)={len(y_true)}, len(per_item_indices)={len(per_item_indices)}")
    # Im Zweifel hart absichern
    assert len(per_item_indices) == len(y_true), "index mapping mismatch"

    fp_json = {
        "project": project,
        "threshold": float(t_test),
        "counts": {
            "fp_class1_0to1": len(fp_list_cls1),
            "fp_class0_1to0": len(fp_list_cls0)
        },
        "aggregation_by_perc": {
            "class1_0to1": {k: v for k, v in by_perc_pair_01.items()},
            "class0_1to0": {k: v for k, v in by_perc_pair_10.items()}
        },
        "aggregation_by_pairs": {
            "0to1": {
                "by_cls_pair_hist": by_cls_pair_01,
                "by_inst_pair_hist": by_inst_pair_01,
                "by_perc_pair_hist": by_perc_pair_01,
                "top_by_cls_pair": _sorted_items(by_cls_pair_01),
                "top_by_inst_pair": _sorted_items(by_inst_pair_01),
                "top_by_perc_pair": _sorted_items(by_perc_pair_01),
            },
            "1to0": {
                "by_cls_pair_hist": by_cls_pair_10,
                "by_inst_pair_hist": by_inst_pair_10,
                "by_perc_pair_hist": by_perc_pair_10,
                "top_by_cls_pair": _sorted_items(by_cls_pair_10),
                "top_by_inst_pair": _sorted_items(by_inst_pair_10),
                "top_by_perc_pair": _sorted_items(by_perc_pair_10),
            }
        },
        "fp_details": {
            "class1_0to1": fp_list_cls1,
            "class0_1to0": fp_list_cls0
        }
    }

    # --- ROBUST SAVE (mit Fehlerausgabe) ---
    fp_json_path = os.path.join(save_path, f"{project}_test_false_positives.json")
    try:
        with open(fp_json_path, "w", encoding="utf-8") as jf:
            json.dump(fp_json, jf, indent=2, ensure_ascii=False)
        print(f"Saved FP details to {fp_json_path}")
    except Exception as e:
        print(f"[SAVE-ERROR] FP JSON: {e}")

    # Excel-Export
    conf_df = pd.DataFrame(cm, index=["True_0", "True_1"], columns=["Pred_0", "Pred_1"])
    per_class = pd.DataFrame({
        "Class": [0, 1], "Support": supp, "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "Precision": prec, "Recall": rec, "F1": f1_cls
    })
    summary = pd.DataFrame({
        "Metric": ["Accuracy", "Precision(macro)", "Recall(macro)", "F1(macro)", "FP_class1(0→1)", "t_star"],
        "Value": [acc, prec_macro, rec_macro, f1_macro, fp_class1, t_test]
    })
    xlsx_path = os.path.join(save_path, f"{project}_test_metrics.xlsx")
    try:
        with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as xw:
            per_class.to_excel(xw, sheet_name="Per_Class", index=False)
            conf_df.to_excel(xw, sheet_name="Confusion_Matrix")
            summary.to_excel(xw, sheet_name="Summary", index=False)
        print(f"Saved test metrics to {xlsx_path}")
    except Exception as e:
        print(f"[SAVE-ERROR] Excel: {e}")

    total_time = time.time() - total_start
    tt_path = os.path.join(save_path, f"{project}_total_training_time.txt")
    try:
        with open(tt_path, "w") as f:
            f.write(f"Total training time: {total_time:.2f} seconds\n")
        print(f"Saved total training time to {tt_path}")
    except Exception as e:
        print(f"[SAVE-ERROR] total_training_time: {e}")
    print(f"Total training time: {total_time:.2f} s")

    return acc, prec_macro, rec_macro, f1_macro


def train_and_evaluate_cnn_xfeat_thresh_new(
    model,
    dataset_with_cv,
    train_loader,      # initialer Loader (nur für steps_per_epoch von OneCycle)
    val_loader,
    test_loader,
    epochs=50,
    learning_rate=3e-4,
    device="cpu",
    save_path="./models",
    project="default",
    only_save_best_model=True,
    cv_info=None,
    fold=None,
    # ---- Threshold-Optionen
    fp0_max_rate=0.006,             # max erlaubte 0->1-Fehlrate (FPR0) auf VAL
    thresh_grid=None,               # z.B. np.linspace(0.5, 0.99, 100)
    thresh_objective="macro_f1",    # "macro_f1" | "f1_class1" | "recall_class1"
    # ---- Hard-Negative-Mining
    max_negatives_per_epoch=1_000_000,
    boost_hard_neg=3.0,
    topk_hard_neg=30_000,
    # ---- Class-Weights
    class_weights_vec=[2.0, 1.0],         # z.B. [8.0, 1.0]; None => auto versuchen
    weight_decay=1e-5,
    grad_accum_steps=4,
    # ---- Loss-Penalty gegen 0→1
    lambda_fp=0.5,          # 0.2–1.0 probieren
    t_neg=0.6,              # negatives p1 unter diese Schwelle drücken
    # ---- TTA (Normal-Hist Rotationen)
    tta_k=0                 # 0 = aus; 8 = 8 Rotationen (empfehlung: 0..8)
):
    """
    Trainiert mit Hard-Negatives und wählt pro Epoche einen Threshold t* auf der Validation,
    der die Bedingung FPR0 <= fp0_max_rate erfüllt und nach thresh_objective optimiert.
    Nutzt EMA-Gewichte + Temperature-Scaling für die Validierung/Test (stabilere p1).
    Speichert Best-Model (nach F1@t*) und exportiert Testmetriken, inkl. FP-Listen als JSON.

    Zusätzlich: JSON-Ausgabe mit False-Positives je Klasse (0→1 & 1→0) inkl. Keys + Aggregation nach 'perc'.
    """
    import os, time, json, gc
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
    from sklearn.metrics import (
        confusion_matrix, accuracy_score,
        precision_score, recall_score, f1_score,
        precision_recall_fscore_support
    )

    # ---------- Utils ----------
    def _parse_key(key):
        p = key.split("_")
        return "_".join(p[:-3]), p[-3], p[-2], int(p[-1])  # class, inst, perc, idx

    def _get_y_mapped_from_pair(p):
        # Dein Mapping, falls global vorhanden; sonst Fallback
        try:
            return globals()["_y_mapped_from_pair"](p)
        except Exception:
            y = p.get("label", None)
            if y is None or y == 0:
                return None
            return 0 if y == 2 else 1

    def _label_from_pair_idx(i):
        ym = _get_y_mapped_from_pair(dataset_with_cv.all_pairs[i])
        return None if ym is None else int(ym)

    def metrics_from_counts(tn, fp, fn, tp):
        # per Klasse und macro
        prec1 = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec1  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f11   = (2*prec1*rec1)/(prec1+rec1) if (prec1+rec1) > 0 else 0.0

        prec0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        rec0  = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f10   = (2*prec0*rec0)/(prec0+rec0) if (prec0+rec0) > 0 else 0.0

        macro_f1 = 0.5 * (f10 + f11)
        return (prec0, rec0, f10), (prec1, rec1, f11), macro_f1

    def pick_threshold(p1, y, grid, max_fp0_rate, objective="macro_f1"):
        # gibt best_t, dict(metrics) zurück
        if grid is None or len(grid) == 0:
            grid = np.linspace(0.5, 0.99, 100)

        best = None
        best_key = None

        for t in grid:
            pred = (p1 >= t).astype(np.int32)
            cm = confusion_matrix(y, pred, labels=[0, 1])
            if cm.shape == (2,2):
                tn, fp, fn, tp = int(cm[0,0]), int(cm[0,1]), int(cm[1,0]), int(cm[1,1])
            else:
                tn = fp = fn = tp = 0

            denom = tn + fp
            fpr0 = (fp / denom) if denom > 0 else 0.0

            (p0,r0,f10),(p1c,r1c,f11),macro_f1 = metrics_from_counts(tn, fp, fn, tp)

            if objective == "f1_class1":
                key_primary = f11
            elif objective == "recall_class1":
                key_primary = r1c
            else:
                key_primary = macro_f1

            feasible = (fpr0 <= max_fp0_rate)
            key = (1 if feasible else 0, key_primary, -fpr0, -t)

            cand = {
                "t": float(t),
                "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                "fpr0": float(fpr0),
                "macro_f1": float(macro_f1),
                "p0": float(p0), "r0": float(r0), "f10": float(f10),
                "p1": float(p1c), "r1": float(r1c), "f11": float(f11),
                "feasible": bool(feasible)
            }
            if (best_key is None) or (key > best_key):
                best = cand
                best_key = key

        return best["t"], best

    # ----- EMA -----
    class EMA:
        def __init__(self, model, decay=0.999):
            self.decay=decay
            self.shadow={k: p.detach().clone() for k,p in model.state_dict().items()}
        def update(self, model):
            with torch.no_grad():
                for k,p in model.state_dict().items():
                    self.shadow[k].mul_(self.decay).add_(p.detach(), alpha=1-self.decay)
        def swap_to(self, model):
            self.backup = {k: p.detach().clone() for k,p in model.state_dict().items()}
            model.load_state_dict(self.shadow, strict=True)
        def swap_back(self, model):
            model.load_state_dict(self.backup, strict=True); del self.backup

    # ----- Temperature Scaling -----
    class TemperatureScaler(nn.Module):
        def __init__(self):
            super().__init__()
            self.log_T = nn.Parameter(torch.zeros(()))  # T=1
        def forward(self, logits):
            T = torch.exp(self.log_T)
            return logits / T

    def fit_temperature_on_logits(logits, y):
        # logits, y: torch tensors on same device
        ce = nn.CrossEntropyLoss()
        scaler = TemperatureScaler().to(logits.device)
        opt = torch.optim.LBFGS(scaler.parameters(), lr=0.1, max_iter=50)

        def closure():
            opt.zero_grad(set_to_none=True)
            loss = ce(scaler(logits), y)
            loss.backward()
            return loss
        opt.step(closure)
        with torch.no_grad():
            T = torch.exp(scaler.log_T).item()
        return scaler, T

    # ----- TTA (Normal-Hist Rotation) -----
    def p1_with_tta_normal(x704, xext, model, device, k=8, temp_scaler=None):
        if k is None or k <= 0:
            logits = model(x704, xext)
            if temp_scaler is not None:
                logits = temp_scaler(logits)
            return torch.softmax(logits, dim=1)[:, 1]
        B = x704.size(0)
        last = x704[:, -64:].view(B, 8, 8)
        p1_accum = torch.zeros(B, device=device)
        for s in range(k):
            rolled = torch.roll(last, shifts=s, dims=1).contiguous().view(B, 64)
            x_cat = torch.cat([x704[:, :640], rolled], dim=1)
            logits = model(x_cat, xext)
            if temp_scaler is not None:
                logits = temp_scaler(logits)
            p1_accum += torch.softmax(logits, dim=1)[:, 1]
        return p1_accum / k

    # ---------- Setup ----------
    os.makedirs(save_path, exist_ok=True)
    device = torch.device(device)
    model = model.to(device)
    amp_enabled = (device.type == "cuda")

    # Class-Weights
    if class_weights_vec is not None:
        class_weights = torch.tensor(class_weights_vec, dtype=torch.float32, device=device)
    else:
        try:
            class_weights = make_ce_weights_from_dataset(dataset_with_cv, alpha_neg=1.6, device=device)
        except Exception:
            class_weights = torch.tensor([1.0, 1.0], dtype=torch.float32, device=device)
    print("[CE weights]", class_weights.tolist())

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    steps_per_epoch = max(1, len(train_loader))
    onecycle = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=max(learning_rate, 3e-4),
        epochs=epochs, steps_per_epoch=steps_per_epoch,
        div_factor=25.0, final_div_factor=1e4
    )
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, threshold=1e-4,
        cooldown=1, min_lr=1e-6
    )
    use_plateau_after = max(4, epochs - 4)

    # EMA
    ema = EMA(model, decay=0.999)

    # AMP
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    # Logging (optional TF)
    try:
        import tensorflow as tf
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass
        writer = tf.summary.create_file_writer(os.path.join(save_path, "logs"))
    except Exception:
        writer = None
        tf = None

    best_info = None
    hard_neg_prev = set()
    base_bs = getattr(train_loader, "batch_size", 64)

    total_start = time.time()

    # ---------- Training Loop ----------
    for epoch in range(epochs):
        # (1) Sampler mit Hard-Negs
        try:
            sampler_posneg, ep_indices = (lambda train_idx, all_pairs: (
                (lambda s, idx: (s, idx))(*(
                    (lambda hard_neg: (  # inner builder
                        (lambda pos_idx, neg_idx: (  # split
                            (lambda neg_sampled:
                                (lambda indices, weights: (
                                    WeightedRandomSampler(
                                        weights=torch.tensor(weights, dtype=torch.float32),
                                        num_samples=len(indices),
                                        replacement=True
                                    ),
                                    indices
                                ))(
                                    (lambda keep_all_positives=True:
                                        (np.concatenate([np.asarray(pos_idx, dtype=np.int64), neg_sampled])
                                         if keep_all_positives else neg_sampled)
                                    )(),
                                    [float(boost_hard_neg) if (idx in hard_neg) else 1.0
                                     for idx in (
                                        np.concatenate([np.asarray(pos_idx, dtype=np.int64), neg_sampled])
                                        if True else neg_sampled
                                     )]
                                )
                            )(
                                (np.random.choice(neg_idx, size=max_negatives_per_epoch, replace=False)
                                 if (max_negatives_per_epoch is not None and len(neg_idx) > max_negatives_per_epoch)
                                 else np.asarray(neg_idx, dtype=np.int64))
                            )
                        ))(
                            [i for i in train_idx if (_get_y_mapped_from_pair(all_pairs[i]) == 1)],
                            [i for i in train_idx if (_get_y_mapped_from_pair(all_pairs[i]) == 0)]
                        )
                    ))(hard_neg_prev)
                ))
            ))(dataset_with_cv.train_idx, dataset_with_cv.all_pairs)
            tr_loader = DataLoader(Subset(dataset_with_cv, ep_indices),
                                   batch_size=base_bs, sampler=sampler_posneg,
                                   num_workers=0, drop_last=False)
        except Exception as e:
            print(f"[Sampler-Fallback] {e}")
            tr_loader = train_loader

        # (2) Train
        model.train()
        running_loss = 0.0
        tn_tr = fp_tr = fn_tr = tp_tr = 0
        optimizer.zero_grad(set_to_none=True)
        accum = 0
        num_batches = max(1, len(tr_loader))

        for b, (x704, xext, labels) in enumerate(tr_loader, start=1):
            x704   = x704.to(device, non_blocking=True)
            xext   = xext.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits = model(x704, xext)
                ce = criterion(logits, labels)
                with torch.no_grad():
                    p1 = torch.softmax(logits, dim=1)[:, 1]
                neg_mask = (labels == 0)
                penalty = torch.relu(p1 - t_neg)[neg_mask].mean() if neg_mask.any() else torch.tensor(0.0, device=device)
                loss = ce + lambda_fp * penalty

            scaler.scale(loss / max(1, grad_accum_steps)).backward()
            accum += 1

            # Optimizer-Step
            if (accum == grad_accum_steps) or (b == num_batches):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                onecycle.step()
                ema.update(model)
                accum = 0

            running_loss += loss.item()

            # Online-Counts (Argmax, nur Gefühl)
            with torch.no_grad():
                pred = logits.argmax(1)
                cm = confusion_matrix(labels.detach().cpu().numpy(),
                                      pred.detach().cpu().numpy(), labels=[0, 1])
                if cm.shape == (2,2):
                    tn_tr += int(cm[0,0]); fp_tr += int(cm[0,1])
                    fn_tr += int(cm[1,0]); tp_tr += int(cm[1,1])

            del logits, loss, ce, p1, penalty, pred
            if torch.cuda.is_available() and (b % 100 == 0):
                torch.cuda.empty_cache()

        train_total = tn_tr + fp_tr + fn_tr + tp_tr
        train_acc = (tn_tr + tp_tr) / max(1, train_total)
        (_, _, f10_t), (_, _, f11_t), macro_f1_t = metrics_from_counts(tn_tr, fp_tr, fn_tr, tp_tr)

        # (3) Validation – EMA + TempScaling + (optional) TTA
        model.eval()
        ema.swap_to(model)  # EMA-Gewichte aktiv

        val_loss = 0.0
        # Für TempScaling: rohe Logits sammeln
        logits_list, y_list = [], []

        with torch.no_grad():
            for (x704, xext, labels) in val_loader:
                x704   = x704.to(device, non_blocking=True)
                xext   = xext.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                # rohen Logits für TempScaling sammeln (ohne TTA!)
                logits = model(x704, xext)
                logits_list.append(logits.detach())
                y_list.append(labels.detach())

                # Loss ohne Penalty (klassische CE)
                val_loss += criterion(logits, labels).item()

        # Temperature scaling fitten
        logits_all = torch.cat(logits_list) if len(logits_list) else torch.empty(0, 2, device=device)
        y_all      = torch.cat(y_list) if len(y_list) else torch.empty(0, dtype=torch.long, device=device)
        temp_scaler, temp_T = (fit_temperature_on_logits(logits_all, y_all) if logits_all.numel() else (None, 1.0))

        # p1 für Thresholdsuche: mit TempScaling, ohne TTA (schneller)
        with torch.no_grad():
            p1_val_all = []
            y_val_all  = []
            for (x704, xext, labels) in val_loader:
                x704 = x704.to(device, non_blocking=True)
                xext = xext.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = model(x704, xext)
                if temp_scaler is not None:
                    logits = temp_scaler(logits)
                p1 = torch.softmax(logits, dim=1)[:, 1]
                p1_val_all.append(p1.detach().cpu().numpy())
                y_val_all.append(labels.detach().cpu().numpy())

        p1_val = np.concatenate(p1_val_all) if len(p1_val_all) else np.array([], dtype=np.float32)
        y_val  = np.concatenate(y_val_all)  if len(y_val_all)  else np.array([], dtype=np.int64)

        # Threshold-Suche
        grid = (np.linspace(0.5, 0.99, 100) if thresh_grid is None else np.asarray(thresh_grid, dtype=np.float32))
        t_star, tinfo = pick_threshold(p1=p1_val, y=y_val, grid=grid,
                                       max_fp0_rate=fp0_max_rate, objective=thresh_objective)

        # Val Metrik @ t*
        pred_t = (p1_val >= t_star).astype(np.int32)
        cm_t = confusion_matrix(y_val, pred_t, labels=[0, 1])
        if cm_t.shape == (2,2):
            tn_s, fp_s, fn_s, tp_s = int(cm_t[0,0]), int(cm_t[0,1]), int(cm_t[1,0]), int(cm_t[1,1])
        else:
            tn_s = fp_s = fn_s = tp_s = 0
        (p0_s, r0_s, f10_s), (p1_s, r1_s, f11_s), macro_f1_s = metrics_from_counts(tn_s, fp_s, fn_s, tp_s)
        val_acc_s = (tn_s + tp_s) / max(1, (tn_s + fp_s + fn_s + tp_s))
        fp01_val = fp_s

        # Prints
        print(f"\n[VAL Threshold-Search] fp0_max_rate={fp0_max_rate:.4f} | best t*={t_star:.4f} | feasible={tinfo['feasible']} | fpr0={tinfo['fpr0']:.5f}")
        print(f"  Class 0 -> P={p0_s:.4f} | R={r0_s:.4f} | F1={f10_s:.4f}")
        print(f"  Class 1 -> P={p1_s:.4f} | R={r1_s:.4f} | F1={f11_s:.4f}")
        print(f"  macroF1(t*)={macro_f1_s:.4f} | crit.errors 0→1 = {fp01_val}")
        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1}/{epochs} | lr={cur_lr:.6g}")
        if epoch + 1 >= use_plateau_after:
            plateau.step(macro_f1_s)

        # Logs/Files
        with open(os.path.join(save_path, f"{project}_train_loss.txt"), "a") as f:
            f.write(f"{epoch+1}; {running_loss/max(1,num_batches):.4f}\n")
        with open(os.path.join(save_path, f"{project}_train_accuracy.txt"), "a") as f:
            f.write(f"{epoch+1}; {train_acc:.4f}\n")
        with open(os.path.join(save_path, f"{project}_train_f1.txt"), "a") as f:
            f.write(f"{macro_f1_t:.4f}\n")

        val_batches = max(1, len(val_loader))
        with open(os.path.join(save_path, f"{project}_val_loss.txt"), "a") as f:
            f.write(f"{epoch+1}; {val_loss/val_batches:.4f}\n")
        with open(os.path.join(save_path, f"{project}_val_accuracy.txt"), "a") as f:
            f.write(f"{epoch+1}; {val_acc_s:.4f}\n")
        with open(os.path.join(save_path, f"{project}_val_f1.txt"), "a") as f:
            f.write(f"{epoch+1}; {macro_f1_s:.4f}\n")

        print(
            f"Epoch {epoch+1}/{epochs}, "
            f"TrainLoss: {running_loss/max(1,num_batches):.4f}, "
            f"TrainAcc: {train_acc:.4f}, TrainF1: {macro_f1_t:.4f}, "
            f"ValLoss: {val_loss/val_batches:.4f}, ValAcc@t*: {val_acc_s:.4f}, ValF1@t*: {macro_f1_s:.4f}"
        )

        if writer is not None:
            with writer.as_default():
                tf.summary.scalar("Loss/train", running_loss/max(1,num_batches), step=epoch)
                tf.summary.scalar("Accuracy/train", train_acc, step=epoch)
                tf.summary.scalar("F1/train_macro", macro_f1_t, step=epoch)
                tf.summary.scalar("Loss/val", val_loss/val_batches, step=epoch)
                tf.summary.scalar("Accuracy/val@t*", val_acc_s, step=epoch)
                tf.summary.scalar("F1/val_macro@t*", macro_f1_s, step=epoch)
                tf.summary.scalar("Val/t_star", t_star, step=epoch)
                tf.summary.scalar("Val/FPR0@t*", tinfo["fpr0"], step=epoch)

        # Best-Model sichern (inkl. T*, Temp)
        if (best_info is None) or (macro_f1_s > best_info["val_macro_f1_t"]):
            best_info = {
                "epoch": epoch+1,
                "val_macro_f1_t": float(macro_f1_s),
                "val_t_star": float(t_star),
                "val_fpr0_t": float(tinfo["fpr0"]),
                "objective": thresh_objective,
                "temp_T": float(temp_T)
            }
            best_model_path = os.path.join(save_path, f"{project}_best_model.pth")
            torch.save({
                "epoch": epoch+1,
                "state_dict": model.state_dict(),   # aktuelle Gewichte (nicht EMA)
                "ema_state_dict": ema.shadow,       # EMA Gewichte für Production/Eval
                "t_star": float(t_star),
                "temp_T": float(temp_T)
            }, best_model_path)
            with open(os.path.join(save_path, f"{project}_best_model_info.json"), "w") as jf:
                json.dump({**best_info, "model_save_path": best_model_path}, jf, indent=2)
            print(f"New best model @ epoch {epoch+1}: macroF1(t*)={macro_f1_s:.4f}, t*={t_star:.4f}")

        if not only_save_best_model:
            ep_path = os.path.join(save_path, f"{project}_model_epoch_{epoch+1}_valF1t_{macro_f1_s:.4f}.pth")
            torch.save(model.state_dict(), ep_path)

        ema.swap_back(model)  # zurück zu Traingewichten

        # (4) Hard-Negative Mining für nächste Epoche (ohne TTA; schnell)
        try:
            model.eval()
            ema.swap_to(model)
            train_indices = getattr(dataset_with_cv, "train_idx", list(range(len(dataset_with_cv))))
            neg_indices = [i for i in train_indices if _label_from_pair_idx(i) == 0]
            if len(neg_indices) == 0:
                hard_neg_prev = set()
            else:
                neg_loader = DataLoader(Subset(dataset_with_cv, neg_indices),
                                        batch_size=base_bs, shuffle=False, num_workers=0, drop_last=False)
                scores, buf = [], []
                offset = 0
                with torch.no_grad():
                    for (x704, xext, labels) in neg_loader:
                        x704 = x704.to(device, non_blocking=True)
                        xext = xext.to(device, non_blocking=True)
                        p1 = torch.softmax(model(x704, xext), dim=1)[:, 1].detach().cpu().numpy()
                        n = len(p1)
                        batch_ids = neg_indices[offset:offset+n]
                        offset += n
                        scores.append(p1); buf.extend(batch_ids)
                if len(scores):
                    s = np.concatenate(scores)
                    K = min(int(topk_hard_neg), len(s))
                    topk = np.argpartition(s, -K)[-K:]
                    hard_neg_prev = set(int(buf[i]) for i in topk)
                else:
                    hard_neg_prev = set()
            ema.swap_back(model)
            print(f"[HardNeg] next epoch: {len(hard_neg_prev)} boosted (×{boost_hard_neg})")
        except Exception as e:
            ema.swap_back(model)
            print(f"[HardNeg] skipped: {e}")
            hard_neg_prev = set()

        # Cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    # ---------- Test ----------
    print("\nStart Test Loop")

    # Lade bestes T* (falls nicht da: fallback)
    t_test = float(best_info["val_t_star"]) if best_info is not None else 0.5
    temp_T = float(best_info.get("temp_T", 1.0)) if best_info is not None else 1.0
    temp_scaler_test = TemperatureScaler().to(device)
    with torch.no_grad():
        temp_scaler_test.log_T.copy_(torch.log(torch.tensor(temp_T, device=device)))

    # EMA für Test
    ema.swap_to(model)
    model.eval()

    y_true, y_pred, y_p1 = [], [], []
    # Um FP-Keys zu loggen: ordne jede Batchposition einem globalen Testindex zu
    test_indices_seq = list(getattr(dataset_with_cv, "test_idx", list(range(len(dataset_with_cv)))))
    ptr = 0
    per_item_indices = []

    with torch.no_grad():
        for (x704, xext, labels) in test_loader:
            x704 = x704.to(device, non_blocking=True)
            xext = xext.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # p1 mit TTA + TempScaling
            p1 = p1_with_tta_normal(x704, xext, model, device,
                                    k=int(tta_k), temp_scaler=temp_scaler_test)

            pred = (p1 >= t_test).long()

            bs = labels.size(0)
            per_item_indices.extend(test_indices_seq[ptr:ptr+bs])
            ptr += bs

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(pred.cpu().numpy().tolist())
            y_p1.extend(p1.cpu().numpy().tolist())

    ema.swap_back(model)

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro   = f1_score(y_true, y_pred, average="macro", zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TP = np.diag(cm); FP = cm.sum(axis=0) - TP; FN = cm.sum(axis=1) - TP
    TN = cm.sum() - (TP + FP + FN)

    prec, rec, f1_cls, supp = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )

    print(f"Test @ t*={t_test:.4f}")
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"Precision (macro): {prec_macro:.4f}")
    print(f"Recall (macro):    {rec_macro:.4f}")
    print(f"F1-Score (macro):  {f1_macro:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("Class-wise metrics (TEST):")
    print(f"  Class 0 -> P={prec[0]:.4f} | R={rec[0]:.4f} | F1={f1_cls[0]:.4f} | n={supp[0]}")
    print(f"  Class 1 -> P={prec[1]:.4f} | R={rec[1]:.4f} | F1={f1_cls[1]:.4f} | n={supp[1]}")
    print(f"Kritische Fehler (0→1): {int(cm[0,1])}")

    # ---------- FP-JSON: Keys + Aggregation nach 'perc' ----------
    # FP für Klasse 1: true=0, pred=1 (kritisch)
    # FP für Klasse 0: true=1, pred=0
    fp_list_cls1 = []  # 0->1
    fp_list_cls0 = []  # 1->0
    perc_hist_cls1 = {}
    perc_hist_cls0 = {}

    for pos, (yt, yp, p1_val, ds_idx) in enumerate(zip(y_true, y_pred, y_p1, per_item_indices)):
        if ds_idx is None:
            continue
        pair = dataset_with_cv.all_pairs[ds_idx]
        # Keys
        k_ref  = pair.get("esf_ref", "")
        k_scan = pair.get("esf_scan", "")
        try:
            cls_r, inst_r, perc_r, idx_r = _parse_key(k_ref)
            cls_s, inst_s, perc_s, idx_s = _parse_key(k_scan)
        except Exception:
            cls_r=inst_r=perc_r=idx_r=cls_s=inst_s=perc_s=idx_s=None

        entry = {
            "ds_index": int(ds_idx),
            "true": int(yt),
            "pred": int(yp),
            "p1": float(p1_val),
            "threshold": float(t_test),
            "esf_ref": k_ref,
            "esf_scan": k_scan,
            "parsed_ref":  {"cls": cls_r, "inst": inst_r, "perc": perc_r, "idx": idx_r},
            "parsed_scan": {"cls": cls_s, "inst": inst_s, "perc": perc_s, "idx": idx_s},
        }

        if yt == 0 and yp == 1:
            fp_list_cls1.append(entry)
            if perc_r is not None:
                perc_hist_cls1[perc_r] = perc_hist_cls1.get(perc_r, 0) + 1
        elif yt == 1 and yp == 0:
            fp_list_cls0.append(entry)
            if perc_r is not None:
                perc_hist_cls0[perc_r] = perc_hist_cls0.get(perc_r, 0) + 1

    fp_json = {
        "project": project,
        "threshold": float(t_test),
        "temp_T": float(temp_T),
        "counts": {
            "fp_class1_0to1": len(fp_list_cls1),
            "fp_class0_1to0": len(fp_list_cls0)
        },
        "aggregation_by_perc": {
            "class1_0to1": perc_hist_cls1,
            "class0_1to0": perc_hist_cls0
        },
        "fp_details": {
            "class1_0to1": fp_list_cls1,
            "class0_1to0": fp_list_cls0
        }
    }
    fp_json_path = os.path.join(save_path, f"{project}_test_false_positives.json")
    with open(fp_json_path, "w", encoding="utf-8") as jf:
        json.dump(fp_json, jf, indent=2, ensure_ascii=False)
    print(f"Saved FP details to {fp_json_path}")

    # Excel-Export wie gehabt
    conf_df = pd.DataFrame(cm, index=["True_0","True_1"], columns=["Pred_0","Pred_1"])
    per_class = pd.DataFrame({
        "Class":[0,1], "Support":supp, "TP":TP, "FP":FP, "FN":FN, "TN":TN,
        "Precision":prec, "Recall":rec, "F1":f1_cls
    })
    summary = pd.DataFrame({
        "Metric": ["Accuracy", "Precision(macro)", "Recall(macro)", "F1(macro)",
                   "FP_class1(0→1)", "FP_class0(1→0)", "t_star", "temp_T"],
        "Value":  [acc, prec_macro, rec_macro, f1_macro, int(cm[0,1]), int(cm[1,0]), t_test, temp_T]
    })
    xlsx_path = os.path.join(save_path, f'{project}_test_metrics.xlsx')
    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as xw:
        per_class.to_excel(xw, sheet_name="Per_Class", index=False)
        conf_df.to_excel(xw, sheet_name="Confusion_Matrix")
        summary.to_excel(xw, sheet_name="Summary", index=False)
    print(f"Saved test metrics to {xlsx_path}")

    total_time = time.time() - total_start
    with open(os.path.join(save_path, f"{project}_total_training_time.txt"), "w") as f:
        f.write(f"Total training time: {total_time:.2f} seconds\n")
    print(f"Total training time: {total_time:.2f} s")

    return acc, prec_macro, rec_macro, f1_macro


def train_and_evaluate_cnn_xfeat(
    model,
    dataset_with_cv,
    train_loader,
    val_loader,
    test_loader,
    epochs=50,
    learning_rate=0.001,
    device="cpu",
    save_path="./models",
    project="default",
    only_save_best_model=False,
    cv_info=None,
    fold=None,
):
    import os, time, json, gc
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
    from sklearn.metrics import (
        confusion_matrix,
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        precision_recall_fscore_support,
    )
    import pandas as pd

    # --- optional: RAM-Monitor (falls psutil vorhanden)
    try:
        import psutil
        process = psutil.Process(os.getpid())
    except Exception:
        process = None

    # ---- TensorFlow-Logger optional (wie bei dir)
    try:
        import tensorflow as tf
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass
    except Exception:
        tf = None

    # ---- Helper: Metriken aus 2x2-Counts
    def metrics_from_counts(tn, fp, fn, tp, average="macro"):
        prec1 = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec1  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f11   = (2*prec1*rec1)/(prec1+rec1) if (prec1+rec1) > 0 else 0.0
        prec0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        rec0  = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f10   = (2*prec0*rec0)/(prec0+rec0) if (prec0+rec0) > 0 else 0.0
        sup0 = tn + fp
        sup1 = tp + fn
        if average == "weighted":
            total = sup0 + sup1
            w_f1  = (f10*sup0 + f11*sup1)/total if total > 0 else 0.0
            w_p   = (prec0*sup0 + prec1*sup1)/total if total > 0 else 0.0
            w_r   = (rec0*sup0 + rec1*sup1)/total if total > 0 else 0.0
            return (w_p, w_r, w_f1), (prec0, rec0, f10, sup0), (prec1, rec1, f11, sup1)
        else:  # macro
            m_f1 = (f10 + f11)/2.0
            m_p  = (prec0 + prec1)/2.0
            m_r  = (rec0 + rec1)/2.0
            return (m_p, m_r, m_f1), (prec0, rec0, f10, sup0), (prec1, rec1, f11, sup1)

    # ---- Label-Helper (wie in deiner Sampler-Logik)
    def _label_from_pair_idx(i):
        # Erwartet: dataset_with_cv.all_pairs und _y_mapped_from_pair existieren.
        ym = _y_mapped_from_pair(dataset_with_cv.all_pairs[i])
        return None if ym is None else int(ym)

    # ---- Sampler mit Hard-Negatives
    def make_epoch_sampler(
        train_idx,
        all_pairs,
        max_negatives=1_000_000,
        keep_all_positives=True,
        hard_neg=None,
        boost=8.0
    ):
        hard_neg = hard_neg or set()
        pos_idx, neg_idx = [], []
        for i in train_idx:
            ym = _y_mapped_from_pair(all_pairs[i])
            if ym is None:
                continue
            if ym == 1:
                pos_idx.append(i)
            else:
                neg_idx.append(i)

        # Negatives subsamplen
        if max_negatives is not None and len(neg_idx) > max_negatives:
            neg_sampled = np.random.choice(neg_idx, size=max_negatives, replace=False)
        else:
            neg_sampled = np.asarray(neg_idx, dtype=np.int64)

        # Positives behalten
        if keep_all_positives:
            indices = np.concatenate([np.asarray(pos_idx, dtype=np.int64), neg_sampled])
        else:
            indices = neg_sampled

        # Gewichte: harte 0er boosten
        weights = np.ones(len(indices), dtype=np.float32)
        hard_neg_set = set(hard_neg)
        for k, idx in enumerate(indices):
            if idx in hard_neg_set:
                weights[k] = float(boost)

        sampler = WeightedRandomSampler(
            weights=torch.tensor(weights, dtype=torch.float32),
            num_samples=len(indices),
            replacement=True
        )
        # Indizes shufflen ist hier nicht nötig; Sampler übernimmt Sampling
        return sampler, indices

    # hyperparameter
    target_fpr0 = 0.003  # 0.3%  (stell ein, was du brauchst)
    best_t = 0.5  # wird nach Val gesetzt
    # ---- Model & Optimizer / Loss
    model = model.to(device)

    try:
        class_weights = make_ce_weights_from_dataset(dataset_with_cv, alpha_neg=1.6, device=device)
    except Exception:
        class_weights = None
    print("weights: ", class_weights)
    class_weights = torch.tensor([4.0,1.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.0)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    # OneCycle (deine Wahl)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.98), weight_decay=1e-5)
    onecycle = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3, epochs=epochs, steps_per_epoch=len(train_loader),
        div_factor=25.0, final_div_factor=1e4
    )
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, threshold=1e-4,
        cooldown=1, min_lr=1e-6
    )
    use_plateau_after = epochs - 4
    scheduler = DualMetricScheduler(optimizer, patience=10, mode="max")

    # Logging
    os.makedirs(save_path, exist_ok=True)
    writer = tf.summary.create_file_writer(os.path.join(save_path, "logs")) if tf is not None else None

    # History
    train_losses, val_losses = [], []
    train_f1s,   val_f1s     = [], []
    train_accuracies, val_accuracies = [], []
    best_f1_score = -1.0

    # Accumulation
    mini_batches = True
    accumulation_steps = 4

    # Hard-Negative Mining Hyperparameter
    hard_neg_prev = set()       # wird nach jeder Epoche neu befüllt
    BOOST = 8.0                 # Sampling-Boost für harte 0er
    K_HARD = 100_000             # Top-K negative Trainingsbeispiele nach p1
    orig_bs = getattr(train_loader, "batch_size", 64)

    total_start_time = time.time()

    for epoch in range(epochs):
        # ---------- EPOCH START: Sampler bauen ----------
        if hasattr(dataset_with_cv, "train_idx") and hasattr(dataset_with_cv, "all_pairs"):
            try:
                sampler, ep_indices = make_epoch_sampler(
                    dataset_with_cv.train_idx,
                    dataset_with_cv.all_pairs,
                    max_negatives=1_000_000,
                    keep_all_positives=True,
                    hard_neg=hard_neg_prev,   # booste harte 0er aus letzter Epoche
                    boost=BOOST
                )
                train_loader = DataLoader(
                    Subset(dataset_with_cv, ep_indices),
                    batch_size=orig_bs,
                    sampler=sampler,
                    num_workers=0,
                    drop_last=False
                )
            except Exception as e:
                print(f"Sampler-Fallback: {e}")

        # -------------------- TRAIN --------------------
        model.train()
        epoch_start_time = time.time()
        running_loss = 0.0

        tn_tr = fp_tr = fn_tr = tp_tr = 0
        optimizer.zero_grad(set_to_none=True)
        accum = 0
        num_batches = len(train_loader)

        for b, (x704, xext, labels) in enumerate(train_loader, start=1):
            x704   = x704.to(device, non_blocking=True)
            xext   = xext.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(x704, xext)
            loss = criterion(outputs, labels)

            if mini_batches:
                loss = loss / accumulation_steps

            loss.backward()
            accum += 1

            if (not mini_batches) or (accum == accumulation_steps) or (b == num_batches):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                onecycle.step()
                accum = 0

            running_loss += loss.item() * (accumulation_steps if mini_batches else 1)

            with torch.no_grad():
                preds = outputs.argmax(1)
                cm = confusion_matrix(labels.detach().cpu().numpy(),
                                      preds.detach().cpu().numpy(),
                                      labels=[0, 1])
                tn_tr += int(cm[0,0]); fp_tr += int(cm[0,1])
                fn_tr += int(cm[1,0]); tp_tr += int(cm[1,1])

            del outputs, loss, preds
            if torch.cuda.is_available() and (b % 100 == 0):
                torch.cuda.empty_cache()

        train_total = tn_tr + fp_tr + fn_tr + tp_tr
        train_acc = (tn_tr + tp_tr) / train_total if train_total > 0 else 0.0
        (tP, tR, tF1), _, _ = metrics_from_counts(tn_tr, fp_tr, fn_tr, tp_tr, average="macro")

        train_losses.append(f"{epoch+1}; {running_loss/num_batches:.4f}")
        train_accuracies.append(f"{epoch+1}; {train_acc:.4f}")
        train_f1s.append(f"{epoch+1}; {tF1:.4f}")

        with open(os.path.join(save_path, f"{project}_train_loss.txt"), "a") as f:
            f.write(f"{epoch+1}; {running_loss/num_batches:.4f}\n")
        with open(os.path.join(save_path, f"{project}_train_accuracy.txt"), "a") as f:
            f.write(f"{epoch+1}; {train_acc:.4f}\n")
        with open(os.path.join(save_path, f"{project}_train_f1.txt"), "a") as f:
            f.write(f"{epoch+1}; {tF1:.4f}\n")

        if writer is not None:
            with writer.as_default():
                tf.summary.scalar("Loss/train", running_loss/num_batches, step=epoch)
                tf.summary.scalar("Accuracy/train", train_acc, step=epoch)
                tf.summary.scalar("F1/train", tF1, step=epoch)
                writer.flush()

        # -------------------- VAL --------------------
        model.eval()
        val_loss = 0.0
        tn_v = fp_v = fn_v = tp_v = 0

        with torch.no_grad():
            for (x704, xext, labels) in val_loader:
                x704   = x704.to(device, non_blocking=True)
                xext   = xext.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                outputs = model(x704, xext)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = outputs.argmax(1)
                cm = confusion_matrix(labels.cpu().numpy(), preds.cpu().numpy(), labels=[0, 1])
                tn_v += int(cm[0,0]); fp_v += int(cm[0,1])
                fn_v += int(cm[1,0]); tp_v += int(cm[1,1])

                del outputs, loss, preds

        val_batches = len(val_loader) if len(val_loader) > 0 else 1
        val_acc = (tn_v + tp_v) / max((tn_v + fp_v + fn_v + tp_v), 1)
        (vP, vR, vF1), (p0, r0, f10, sup0), (p1, r1, f11, sup1) = metrics_from_counts(tn_v, fp_v, fn_v, tp_v, average="macro")

        print("Class-wise metrics (VAL):")
        print(f"  Class 0 -> P={p0:.4f} | R={r0:.4f} | F1={f10:.4f} | n={sup0}")
        print(f"  Class 1 -> P={p1:.4f} | R={r1:.4f} | F1={f11:.4f} | n={sup1}")
        print("Class-wise F1 scores:", {"Class 0": f10, "Class 1": f11})

        cur_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1}/{epochs} | lr={cur_lr:.6g}")
        if epoch + 1 >= use_plateau_after:
            plateau.step(vF1)

        val_losses.append(f"{epoch+1}; {val_loss/val_batches:.4f}")
        val_accuracies.append(f"{epoch+1}; {val_acc:.4f}")
        val_f1s.append(f"{epoch+1}; {vF1:.4f}")

        with open(os.path.join(save_path, f"{project}_val_loss.txt"), "a") as f:
            f.write(f"{epoch+1}; {val_loss/val_batches:.4f}\n")
        with open(os.path.join(save_path, f"{project}_val_accuracy.txt"), "a") as f:
            f.write(f"{epoch+1}; {val_acc:.4f}\n")
        with open(os.path.join(save_path, f"{project}_val_f1.txt"), "a") as f:
            f.write(f"{epoch+1}; {vF1:.4f}\n")

        print(
            f"Epoch {epoch+1}/{epochs}, "
            f"Train Loss: {train_losses[-1].split('; ')[1]}, "
            f"Train Accuracy: {train_acc:.4f}, Train F1 Score: {tF1:.4f}, "
            f"Val Loss: {val_losses[-1].split('; ')[1]}, Val Accuracy: {val_acc:.4f}, Val F1 Score: {vF1:.4f}"
        )

        if writer is not None:
            with writer.as_default():
                tf.summary.scalar("Loss/val", val_loss/val_batches, step=epoch)
                tf.summary.scalar("Accuracy/val", val_acc, step=epoch)
                tf.summary.scalar("F1/val", vF1, step=epoch)
                tf.summary.scalar("Val/Class0_Precision", p0, step=epoch)
                tf.summary.scalar("Val/Class0_Recall", r0, step=epoch)
                tf.summary.scalar("Val/Class1_Precision", p1, step=epoch)
                tf.summary.scalar("Val/Class1_Recall", r1, step=epoch)
                writer.flush()

        # Best-Model speichern
        if vF1 > best_f1_score:
            best_f1_score = vF1
            best_model_save_path = os.path.join(save_path, f"{project}_best_model.pth")
            torch.save({"epoch": epoch+1,
                        "f1_score": float(best_f1_score),
                        "state_dict": model.state_dict()}, best_model_save_path)
            print(f"New best model found with F1 Score: {best_f1_score:.4f} at epoch {epoch+1}")

            best_model_info = {
                "epoch": epoch+1,
                "f1_score": float(best_f1_score),
                "model_save_path": best_model_save_path
            }
            with open(os.path.join(save_path, f"{project}_best_model_info.json"), "w") as json_file:
                json.dump(best_model_info, json_file)

        if not only_save_best_model:
            model_save_path = os.path.join(
                save_path, f"{project}_model_epoch_{epoch+1}_acc_{np.round(val_acc,4)}_f1_{np.round(vF1,4)}.pth"
            )
            torch.save(model.state_dict(), model_save_path)

        # ---------- HARD NEGATIVE MINING: NEU BEFÜLLEN ----------
        try:
            model.eval()
            if hasattr(dataset_with_cv, "train_idx") and hasattr(dataset_with_cv, "all_pairs"):
                train_indices = dataset_with_cv.train_idx
            else:
                train_indices = list(range(len(dataset_with_cv)))

            # nur echte Negative (Label==0)
            neg_indices = []
            for i in train_indices:
                lab = _label_from_pair_idx(i)
                if lab is None:
                    continue
                if lab == 0:
                    neg_indices.append(i)

            if len(neg_indices) == 0:
                hard_neg_prev = set()
                print("[HardNeg] keine negativen Trainingsbeispiele.")
            else:
                neg_loader = DataLoader(
                    Subset(dataset_with_cv, neg_indices),
                    batch_size=orig_bs, shuffle=False, num_workers=0, drop_last=False
                )

                yscore = []
                idxbuf = []
                offset = 0
                with torch.no_grad():
                    for (x704, xext, labels) in neg_loader:
                        x704 = x704.to(device, non_blocking=True)
                        xext = xext.to(device, non_blocking=True)
                        logits = model(x704, xext)
                        p1 = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()

                        bs = len(p1)
                        batch_ids = neg_indices[offset: offset + bs]
                        offset += bs

                        yscore.append(p1)
                        idxbuf.extend(batch_ids)

                scores = np.concatenate(yscore) if len(yscore) else np.array([], dtype=np.float32)
                K = min(int(K_HARD), len(scores))
                if K > 0:
                    topk_idx = np.argpartition(scores, -K)[-K:]
                    hard_neg_prev = set(int(idxbuf[i]) for i in topk_idx)
                else:
                    hard_neg_prev = set()

                print(f"[HardNeg] neu markiert: {len(hard_neg_prev)} / {len(neg_indices)} (Boost={BOOST}x)")
        except Exception as e:
            print(f"[HardNeg] übersprungen: {e}")
            hard_neg_prev = set()

        # Epoche Zeit + Speicher
        epoch_time = time.time() - epoch_start_time
        with open(os.path.join(save_path, f"{project}_epoch_times.txt"), "a") as f:
            f.write(f"Epoch {epoch+1}: {epoch_time:.2f} seconds\n")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        if process is not None:
            mem_gb = process.memory_info().rss / (1024 ** 3)
            print(f"[Epoch {epoch + 1}] RAM: {mem_gb:.2f} GB")
        if torch.cuda.is_available():
            print(f"[Epoch {epoch + 1}] VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    total_training_time = time.time() - total_start_time
    with open(os.path.join(save_path, f"{project}_total_training_time.txt"), "w") as f:
        f.write(f"Total training time: {total_training_time:.2f} seconds\n")
    print(f"Total training time: {total_training_time:.2f} seconds")

    # -------------------- TEST --------------------
    print("Start Test Loop")
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for batch_idx, (esf_diff, extra_features, labels) in enumerate(test_loader):
            esf_diff = esf_diff.to(device)
            extra_features = extra_features.to(device)
            labels = labels.to(device)

            outputs = model(esf_diff, extra_features)
            _, predicted = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(predicted.cpu().numpy().tolist())

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    test_accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TP = np.diag(cm)
    FP = cm.sum(axis=0) - TP
    FN = cm.sum(axis=1) - TP
    TN = cm.sum() - (TP + FP + FN)

    prec, rec, f1_cls, supp = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )

    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Precision (macro): {precision_macro:.4f}")
    print(f"Recall (macro):    {recall_macro:.4f}")
    print(f"F1-Score (macro):  {f1_macro:.4f}")
    print("Confusion Matrix:")
    print(cm)

    print("Class-wise metrics (TEST):")
    print(f"  Class 0 -> P={prec[0]:.4f} | R={rec[0]:.4f} | F1={f1_cls[0]:.4f} | n={supp[0]}")
    print(f"  Class 1 -> P={prec[1]:.4f} | R={rec[1]:.4f} | F1={f1_cls[1]:.4f} | n={supp[1]}")

    fp_class1 = cm[0, 1]
    print(f"Kritische Fehler (0→1): {fp_class1}")

    conf_matrix_df = pd.DataFrame(
        cm,
        index=[f"True_{i}" for i in [0, 1]],
        columns=[f"Pred_{i}" for i in [0, 1]]
    )

    metrics = {
        "Class": [0, 1],
        "Support": supp,
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
        "Precision": prec,
        "Recall": rec,
        "F1": f1_cls,
    }
    df_metrics = pd.DataFrame(metrics)

    summary = pd.DataFrame({
        "Metric": ["Accuracy", "Precision(macro)", "Recall(macro)", "F1(macro)", "FP_class1(0→1)"],
        "Value": [test_accuracy, precision_macro, recall_macro, f1_macro, fp_class1]
    })

    excel_path = os.path.join(save_path, f'{project}_test_metrics.xlsx')
    with pd.ExcelWriter(excel_path, engine='xlsxwriter') as xls_writer:
        df_metrics.to_excel(xls_writer, sheet_name='Per_Class', index=False)
        conf_matrix_df.to_excel(xls_writer, sheet_name='Confusion_Matrix')
        summary.to_excel(xls_writer, sheet_name='Summary', index=False)

    print(f"Saved test metrics to {excel_path}")


def train_and_evaluate_cnn_xfeat_back(
    model,
    dataset_with_cv,
    train_loader,
    val_loader,
    test_loader,
    epochs=50,
    learning_rate=0.001,
    device="cpu",
    save_path="./models",
    project="default",
    only_save_best_model=False,
    cv_info=None,
    fold=None,
):
    import os, time, json, gc
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Subset
    from sklearn.metrics import confusion_matrix
    import pandas as pd

    # ---- TensorFlow für Logging nutzen, aber NICHT die GPU belegen (Windows mag das sonst gar nicht)
    try:
        import tensorflow as tf
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass
    except Exception:
        tf = None

    # ---- Helper: Metriken aus 2x2-Counts berechnen (binary, Klassen 0/1)
    def metrics_from_counts(tn, fp, fn, tp, average="macro"):
        # Klasse 1 (positiv)
        prec1 = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec1  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f11   = (2*prec1*rec1)/(prec1+rec1) if (prec1+rec1) > 0 else 0.0
        # Klasse 0 als "positiv" denken (Spiegelung)
        prec0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        rec0  = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f10   = (2*prec0*rec0)/(prec0+rec0) if (prec0+rec0) > 0 else 0.0
        # Supports
        sup0 = tn + fp
        sup1 = tp + fn
        # Aggregationen
        if average == "weighted":
            total = sup0 + sup1
            w_f1  = (f10*sup0 + f11*sup1)/total if total > 0 else 0.0
            w_p   = (prec0*sup0 + prec1*sup1)/total if total > 0 else 0.0
            w_r   = (rec0*sup0 + rec1*sup1)/total if total > 0 else 0.0
            return (w_p, w_r, w_f1), (prec0, rec0, f10, sup0), (prec1, rec1, f11, sup1)
        else:  # 'macro'
            m_f1 = (f10 + f11)/2.0
            m_p  = (prec0 + prec1)/2.0
            m_r  = (rec0 + rec1)/2.0
            return (m_p, m_r, m_f1), (prec0, rec0, f10, sup0), (prec1, rec1, f11, sup1)

    # ---- Model & Optimizer / Loss
    model = model.to(device)

    # Falls du Gewichte möchtest – dein Helper, sonst None
    try:
        class_weights = make_ce_weights_from_dataset(dataset_with_cv, alpha_neg=1.6, device=device)
    except Exception:
        class_weights = None

    print("weights: ", class_weights)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.0)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    # Phase 1: OneCycle
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.98), weight_decay=1e-5)
    onecycle = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3, epochs=epochs, steps_per_epoch=len(train_loader),
        div_factor=25.0, final_div_factor=1e4
    )

    # ... Training: pro BATCH -> optimizer.step(); onecycle.step()

    # Optional Phase 2: Feintuning (letzte k Epochen)
    # Deaktiviere OneCycle wenn fertig. Dann:

    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, threshold=1e-4,
        cooldown=1, min_lr=1e-6
    )
    use_plateau_after = epochs - 4  # letzte 4 Epochen z. B.
    # pro EPOCHE nach Val: plateau.step(val_f1)
    # Dein Scheduler (nimmt val_loss und val_f1)
    scheduler = DualMetricScheduler(optimizer, patience=10, mode="max")

    # Logging
    os.makedirs(save_path, exist_ok=True)
    writer = tf.summary.create_file_writer(os.path.join(save_path, "logs")) if tf is not None else None

    # History
    train_losses, val_losses = [], []
    train_f1s,   val_f1s     = [], []
    train_accuracies, val_accuracies = [], []
    best_f1_score = -1.0

    # Accumulation (optional)
    mini_batches = True
    accumulation_steps = 4

    total_start_time = time.time()

    for epoch in range(epochs):
        # Optional: Epochen-spezifisches Samplen
        if hasattr(dataset_with_cv, "train_idx") and hasattr(dataset_with_cv, "all_pairs"):
            try:
                sampler = make_epoch_sampler(dataset_with_cv.train_idx, dataset_with_cv.all_pairs, max_negatives=1_000_000)
                train_loader = DataLoader(Subset(dataset_with_cv, dataset_with_cv.train_idx),
                                          batch_size=128, sampler=sampler, num_workers=0, drop_last=False)
            except Exception:
                pass  # falls make_epoch_sampler nicht verfügbar ist

        # -------------------- TRAIN --------------------
        model.train()
        epoch_start_time = time.time()
        running_loss = 0.0

        # Streaming-Counts (binary)
        tn_tr = fp_tr = fn_tr = tp_tr = 0

        optimizer.zero_grad(set_to_none=True)
        accum = 0
        num_batches = len(train_loader)

        for b, (x704, xext, labels) in enumerate(train_loader, start=1):
            x704   = x704.to(device, non_blocking=True)
            xext   = xext.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(x704, xext)
            loss = criterion(outputs, labels)

            if mini_batches:
                loss = loss / accumulation_steps

            loss.backward()
            accum += 1

            # Optimizer-Step bei Accum-Full oder letztem Batch
            if (not mini_batches) or (accum == accumulation_steps) or (b == num_batches):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                onecycle.step()
                accum = 0

            # Loss-Tracking in "Originalskala"
            running_loss += loss.item() * (accumulation_steps if mini_batches else 1)

            # Batch-Confusion addieren (streaming, keine Listen!)
            with torch.no_grad():
                preds = outputs.argmax(1)
                cm = confusion_matrix(labels.detach().cpu().numpy(),
                                      preds.detach().cpu().numpy(),
                                      labels=[0, 1])
                # cm: [[tn, fp],[fn, tp]]
                tn_tr += int(cm[0,0]); fp_tr += int(cm[0,1])
                fn_tr += int(cm[1,0]); tp_tr += int(cm[1,1])

            # Speicher entlasten
            del outputs, loss, preds
            if torch.cuda.is_available() and (b % 100 == 0):
                torch.cuda.empty_cache()

        # Train-Metriken aus Counts
        train_total = tn_tr + fp_tr + fn_tr + tp_tr
        train_acc = (tn_tr + tp_tr) / train_total if train_total > 0 else 0.0
        (tP, tR, tF1), _, _ = metrics_from_counts(tn_tr, fp_tr, fn_tr, tp_tr, average="macro")

        train_losses.append(f"{epoch+1}; {running_loss/num_batches:.4f}")
        train_accuracies.append(f"{epoch+1}; {train_acc:.4f}")
        train_f1s.append(f"{epoch+1}; {tF1:.4f}")

        # File-Logging
        with open(os.path.join(save_path, f"{project}_train_loss.txt"), "a") as f:
            f.write(f"{epoch+1}; {running_loss/num_batches:.4f}\n")
        with open(os.path.join(save_path, f"{project}_train_accuracy.txt"), "a") as f:
            f.write(f"{epoch+1}; {train_acc:.4f}\n")
        with open(os.path.join(save_path, f"{project}_train_f1.txt"), "a") as f:
            f.write(f"{epoch+1}; {tF1:.4f}\n")

        if writer is not None:
            with writer.as_default():
                tf.summary.scalar("Loss/train", running_loss/num_batches, step=epoch)
                tf.summary.scalar("Accuracy/train", train_acc, step=epoch)
                tf.summary.scalar("F1/train", tF1, step=epoch)
                writer.flush()

        # -------------------- VAL --------------------
        model.eval()
        val_loss = 0.0
        tn_v = fp_v = fn_v = tp_v = 0

        with torch.no_grad():
            for (x704, xext, labels) in val_loader:
                x704   = x704.to(device, non_blocking=True)
                xext   = xext.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                outputs = model(x704, xext)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = outputs.argmax(1)
                cm = confusion_matrix(labels.cpu().numpy(), preds.cpu().numpy(), labels=[0, 1])
                tn_v += int(cm[0,0]); fp_v += int(cm[0,1])
                fn_v += int(cm[1,0]); tp_v += int(cm[1,1])

                del outputs, loss, preds

        val_batches = len(val_loader) if len(val_loader) > 0 else 1
        val_acc = (tn_v + tp_v) / max((tn_v + fp_v + fn_v + tp_v), 1)
        (vP, vR, vF1), (p0, r0, f10, sup0), (p1, r1, f11, sup1) = metrics_from_counts(tn_v, fp_v, fn_v, tp_v, average="macro")

        print("Class-wise metrics (VAL):")
        print(f"  Class 0 -> P={p0:.4f} | R={r0:.4f} | F1={f10:.4f} | n={sup0}")
        print(f"  Class 1 -> P={p1:.4f} | R={r1:.4f} | F1={f11:.4f} | n={sup1}")
        print("Class-wise F1 scores:", {"Class 0": f10, "Class 1": f11})

        # Scheduler entscheidet nach Loss & F1
        cur_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1}/{epochs} | lr={cur_lr:.6g}")
        # nach VAL und Prints
        if epoch + 1 >= use_plateau_after:
            plateau.step(vF1)  # oder val_loss negativ, je nach Ziel


        val_losses.append(f"{epoch+1}; {val_loss/val_batches:.4f}")
        val_accuracies.append(f"{epoch+1}; {val_acc:.4f}")
        val_f1s.append(f"{epoch+1}; {vF1:.4f}")

        with open(os.path.join(save_path, f"{project}_val_loss.txt"), "a") as f:
            f.write(f"{epoch+1}; {val_loss/val_batches:.4f}\n")
        with open(os.path.join(save_path, f"{project}_val_accuracy.txt"), "a") as f:
            f.write(f"{epoch+1}; {val_acc:.4f}\n")
        with open(os.path.join(save_path, f"{project}_val_f1.txt"), "a") as f:
            f.write(f"{epoch+1}; {vF1:.4f}\n")

        print(
            f"Epoch {epoch+1}/{epochs}, "
            f"Train Loss: {train_losses[-1].split('; ')[1]}, "
            f"Train Accuracy: {train_acc:.4f}, Train F1 Score: {tF1:.4f}, "
            f"Val Loss: {val_losses[-1].split('; ')[1]}, Val Accuracy: {val_acc:.4f}, Val F1 Score: {vF1:.4f}"
        )

        if writer is not None:
            with writer.as_default():
                tf.summary.scalar("Loss/val", val_loss/val_batches, step=epoch)
                tf.summary.scalar("Accuracy/val", val_acc, step=epoch)
                tf.summary.scalar("F1/val", vF1, step=epoch)
                tf.summary.scalar("Val/Class0_Precision", p0, step=epoch)
                tf.summary.scalar("Val/Class0_Recall", r0, step=epoch)
                tf.summary.scalar("Val/Class1_Precision", p1, step=epoch)
                tf.summary.scalar("Val/Class1_Recall", r1, step=epoch)
                writer.flush()

        # Best-Model SPEICHERSCHONEND speichern (keine zweite Kopie im RAM halten)
        if vF1 > best_f1_score:
            best_f1_score = vF1
            best_model_save_path = os.path.join(save_path, f"{project}_best_model.pth")
            torch.save({"epoch": epoch+1,
                        "f1_score": float(best_f1_score),
                        "state_dict": model.state_dict()}, best_model_save_path)
            print(f"New best model found with F1 Score: {best_f1_score:.4f} at epoch {epoch+1}")

            best_model_info = {
                "epoch": epoch+1,
                "f1_score": float(best_f1_score),
                "model_save_path": best_model_save_path
            }
            with open(os.path.join(save_path, f"{project}_best_model_info.json"), "w") as json_file:
                json.dump(best_model_info, json_file)

        if not only_save_best_model:
            model_save_path = os.path.join(
                save_path, f"{project}_model_epoch_{epoch+1}_acc_{np.round(val_acc,4)}_f1_{np.round(vF1,4)}.pth"
            )
            torch.save(model.state_dict(), model_save_path)

        # Epoche Zeit loggen
        epoch_time = time.time() - epoch_start_time
        with open(os.path.join(save_path, f"{project}_epoch_times.txt"), "a") as f:
            f.write(f"Epoch {epoch+1}: {epoch_time:.2f} seconds\n")

        # Speicher aufräumen
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Optional: Peak-Stats
            # print(f"GPU mem now: {torch.cuda.memory_allocated()/1e9:.2f} GB | peak: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
            torch.cuda.reset_peak_memory_stats()
        # Am Ende jeder Epoche
        mem_gb = process.memory_info().rss / (1024 ** 3)
        print(f"[Epoch {epoch + 1}] RAM: {mem_gb:.2f} GB")

        if torch.cuda.is_available():
            print(f"[Epoch {epoch + 1}] VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    total_training_time = time.time() - total_start_time
    with open(os.path.join(save_path, f"{project}_total_training_time.txt"), "w") as f:
        f.write(f"Total training time: {total_training_time:.2f} seconds\n")
    print(f"Total training time: {total_training_time:.2f} seconds")

    # -------------------- TEST --------------------
    print("Start Test Loop")
    # --- Test ---
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for batch_idx, (esf_diff, extra_features, labels) in enumerate(test_loader):
            esf_diff = esf_diff.to(device)
            extra_features = extra_features.to(device)
            labels = labels.to(device)

            outputs = model(esf_diff, extra_features)
            _, predicted = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(predicted.cpu().numpy().tolist())

    # numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Grundmetriken (macro, wie bei dir)
    test_accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # Confusion-Matrix + per-class P/R/F1
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TP = np.diag(cm)
    FP = cm.sum(axis=0) - TP
    FN = cm.sum(axis=1) - TP
    TN = cm.sum() - (TP + FP + FN)

    # per-class via sklearn (bequem & robust)
    prec, rec, f1_cls, supp = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )

    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Precision (macro): {precision_macro:.4f}")
    print(f"Recall (macro):    {recall_macro:.4f}")
    print(f"F1-Score (macro):  {f1_macro:.4f}")
    print("Confusion Matrix:")
    print(cm)

    print("Class-wise metrics (TEST):")
    print(f"  Class 0 -> P={prec[0]:.4f} | R={rec[0]:.4f} | F1={f1_cls[0]:.4f} | n={supp[0]}")
    print(f"  Class 1 -> P={prec[1]:.4f} | R={rec[1]:.4f} | F1={f1_cls[1]:.4f} | n={supp[1]}")

    # (optional) Kritische Fehler explizit ausgeben: 0->1 = FP für Klasse 1
    fp_class1 = cm[0, 1]
    print(f"Kritische Fehler (0→1): {fp_class1}")

    # ---- Excel speichern (NEUER Name für Writer! Kein flush hier!) ----
    conf_matrix_df = pd.DataFrame(
        cm,
        index=[f"True_{i}" for i in [0, 1]],
        columns=[f"Pred_{i}" for i in [0, 1]]
    )

    metrics = {
        "Class": [0, 1],
        "Support": supp,
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
        "Precision": prec,
        "Recall": rec,
        "F1": f1_cls,
    }
    df_metrics = pd.DataFrame(metrics)

    summary = pd.DataFrame({
        "Metric": ["Accuracy", "Precision(macro)", "Recall(macro)", "F1(macro)", "FP_class1(0→1)"],
        "Value": [test_accuracy, precision_macro, recall_macro, f1_macro, fp_class1]
    })

    excel_path = os.path.join(save_path, f'{project}_test_metrics.xlsx')
    with pd.ExcelWriter(excel_path, engine='xlsxwriter') as xls_writer:
        df_metrics.to_excel(xls_writer, sheet_name='Per_Class', index=False)
        conf_matrix_df.to_excel(xls_writer, sheet_name='Confusion_Matrix')
        summary.to_excel(xls_writer, sheet_name='Summary', index=False)

    print(f"Saved test metrics to {excel_path}")



def get_label_of_idx(ds, i):
    # Passe das an deine Dataset-API an.
    if hasattr(ds, "labels"):
        return int(ds.labels[i])
    if hasattr(ds, "targets"):
        return int(ds.targets[i])
    # Fallback: einmal item holen (langsam, aber robust)
    item = ds[i]
    return int(item[-1])  # erwartet, dass letztes Element das Label ist

def make_weights(train_idx, hard_neg_set, boost=5.0):
    # Basisgewicht 1.0 für alle
    w = np.ones(len(train_idx), dtype=np.float32)
    # Indizes, die in hard_neg_set liegen, bekommen Boost
    pos = [k for k, idx in enumerate(train_idx) if idx in hard_neg_set]
    if pos:
        w[pos] = boost
    return torch.tensor(w, dtype=torch.float32)


def compute_pos_weight_from_loader(loader, device="cpu"):
    all_labels = []
    for _, _, labels in loader:
        all_labels.append(labels)

    # Alle Labels zu einem Tensor zusammenführen
    all_labels = torch.cat(all_labels)

    # Annahme: 0 = negativ, 1 = positiv
    N_pos = (all_labels == 1).sum().item()
    N_neg = (all_labels == 0).sum().item()

    # pos_weight für BCEWithLogitsLoss
    pos_weight = torch.tensor([N_neg / N_pos], dtype=torch.float32, device=device)

    return N_pos, N_neg, pos_weight

def train_and_evaluate_cnn_5_channels(model, dataset_with_cv, train_loader, val_loader, test_loader, epochs=50, learning_rate=0.001, device = "cpu", save_path='./models', project = "default", only_save_best_model = False, cv_info=None, fold = None):
    # Move the model to the GPU (if available)
    model = model.to(device)

    #criterion = nn.CrossEntropyLoss(label_smoothing=0.1) # Output shape: [B, 3], Labels: LongTensor [B]
    #class_weights = torch.tensor([1.0, 1.5, 0.5]).to(device)
    #class_weights = torch.tensor([1.0, 1.0, 1.0]).to(device)

    #criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    # Optional: Class Imbalance ausgleichen
    N_pos, N_neg, pos_weight= compute_pos_weight_from_loader(train_loader, device="cpu")
    pos_weight = torch.tensor([N_neg / N_pos], device=device)  # berechne aus Dataset
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    #criterion = FocalLoss(alpha=[0.25, 0.75], gamma=2.0)
    #criterion = nn.BCEWithLogitsLoss() # binärer fall # Output muss shape [B, 1] sein → dann: labels = labels.float().unsqueeze(1)

    average_metric = "macro" # 'macro' , 'weighted

    # Focal Loss initialisieren
    #criterion = FocalLoss(alpha=0.25, gamma=2)

    #optimizer = optim.Adam(model.parameters(), lr=learning_rate)#,  weight_decay=0.0001)#, weight_decay=0.01) #optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9) #optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    #optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    scheduler = DualMetricScheduler(optimizer, patience=10, mode="max")

    #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=setup["learning_rate"], steps_per_epoch=len(train_loader),epochs=setup["epochs"], pct_start=0.1)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    # Ensure save directory exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # Definiere das Log-Verzeichnis
    logdir = os.path.join(save_path, 'logs')
    writer = tf.summary.create_file_writer(logdir)
    # tensorboard --logdir= logdir

    # Initialize lists to store history
    train_losses = []
    val_losses = []
    train_f1s = []
    val_f1s = []
    train_accuracies = []
    val_accuracies = []
    best_f1_score = 0
    best_model = None
    mini_batches = True
    accumulation_steps = 4  # Simuliert größere Batchgröße

    total_start_time = time.time()

    # Initialisiere die Subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    # Training Loop
    for epoch in range(epochs):
        epoch_start_time = time.time()  # Zeitmessung für die Epoche starten
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        all_labels_train = []
        all_predictions_train = []


        for i, (esf_diff, extra_features, labels) in enumerate(train_loader):
            # Move inputs and labels to GPU (if available)
            esf_diff = esf_diff.to(device)
            labels = labels.to(device).float()  # <- wichtig!
            # Vorwärtsdurchlauf
            outputs = model(esf_diff).squeeze(1)
            loss = criterion(outputs, labels.float())
            #loss = combined_loss_function(outputs, labels, alpha=0.5)

            if mini_batches == True:
                loss = loss / accumulation_steps  # Loss normalisieren für Accumulation

            # Rückwärtsdurchlauf
            loss.backward()

            if mini_batches:
                if (i + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            else:
                optimizer.step()
                optimizer.zero_grad()

            # Laufenden Verlust berechnen
            running_loss += loss.item() * (accumulation_steps if mini_batches else 1)



            # Compute training accuracy
            #_, predicted = torch.max(outputs, 1)
            probs = torch.sigmoid(outputs)  # wandelt logits in Wahrscheinlichkeiten
            predicted = (probs > 0.5).long()  # 0 oder 1 als Prediction
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            # Collect all labels and predictions for F1 score calculation
            all_labels_train.extend(labels.cpu().numpy())
            all_predictions_train.extend(predicted.cpu().numpy())

        train_accuracy = correct_train / total_train
        train_f1 = f1_score(all_labels_train, all_predictions_train, average=average_metric)

        train_losses.append(f"{epoch + 1}; {running_loss / len(train_loader):.4f}")
        train_accuracies.append(f"{epoch + 1}; {train_accuracy:.4f}")
        train_f1s.append(f"{epoch + 1}; {train_f1:.4f}")

        # Update train history files
        with open(os.path.join(save_path, f'{project}_train_loss.txt'), 'a') as f:
            f.write(f"{epoch + 1}; {running_loss / len(train_loader):.4f}\n")

        with open(os.path.join(save_path, f'{project}_train_accuracy.txt'), 'a') as f:
            f.write(f"{epoch + 1}; {train_accuracy:.4f}\n")

        with open(os.path.join(save_path, f'{project}_train_f1.txt'), 'a') as f:
            f.write(f"{epoch + 1}; {train_f1:.4f}\n")

        # Log metrics to TensorBoard
        with writer.as_default():
            tf.summary.scalar('Loss/train', running_loss / len(train_loader), step=epoch)
            tf.summary.scalar('Accuracy/train', train_accuracy, step=epoch)
            tf.summary.scalar('F1/train', train_f1, step=epoch)
            writer.flush()  # Sicherstellen, dass die Daten geschrieben werden



        # Validation Loop
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for (esf_diff, extra_features, labels) in val_loader:
                esf_diff = esf_diff.to(device)
                labels = labels.to(device)

                outputs = model(esf_diff).squeeze(1)  # [B]
                loss = criterion(outputs, labels.float())
                val_loss += loss.item()

                probs = torch.sigmoid(outputs)  # [B]
                predicted = (probs > 0.5).long()  # [B]
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        val_accuracy = correct_val / total_val

        #val_f1 = f1_score(all_labels, all_predictions, average='weighted')
        val_f1 = f1_score(all_labels, all_predictions, average=average_metric)
        # F1 scores for each label
        class_f1_scores = f1_score(all_labels, all_predictions, average=None)
        # Store F1 scores in a dictionary
        f1_scores_per_class = {f"Class {i}": f1 for i, f1 in enumerate(class_f1_scores)}

        # Example: Save to a file or log
        print("Class-wise F1 scores:", f1_scores_per_class)

        #scheduler.step(val_loss)
        scheduler.step(val_loss, val_f1)

        val_losses.append(f"{epoch + 1}; {val_loss / len(val_loader):.4f}")
        val_accuracies.append(f"{epoch + 1}; {val_accuracy:.4f}")
        val_f1s.append(f"{epoch + 1}; {val_f1:.4f}")



        # Update validation history files
        with open(os.path.join(save_path, f'{project}_val_loss.txt'), 'a') as f:
            f.write(f"{epoch + 1}; {val_loss / len(val_loader):.4f}\n")

        with open(os.path.join(save_path, f'{project}_val_accuracy.txt'), 'a') as f:
            f.write(f"{epoch + 1}; {val_accuracy:.4f}\n")

        with open(os.path.join(save_path, f'{project}_val_f1.txt'), 'a') as f:
            f.write(f"{epoch + 1}; {val_f1:.4f}\n")

        # Print metrics
        print(
            f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_losses[-1].split("; ")[1]}, Train Accuracy: {train_accuracy:.4f}, Train F1 Score: {train_f1:.4f}, Val Loss: {val_losses[-1].split("; ")[1]}, Val Accuracy: {val_accuracy:.4f}, Val F1 Score: {val_f1:.4f}')

        # Log validation metrics to TensorBoard
        with writer.as_default():
            tf.summary.scalar('Loss/val', val_loss / len(val_loader), step=epoch)
            tf.summary.scalar('Accuracy/val', val_accuracy, step=epoch)
            tf.summary.scalar('F1/val', val_f1, step=epoch)
            writer.flush()

        # Check if this is the best model based on F1 score
        if val_f1 > best_f1_score:
            best_f1_score = val_f1
            best_model = {
                'epoch': epoch + 1,
                'f1_score': best_f1_score,
                'model_state_dict': model.state_dict()
            }

            # Save the best model
            best_model_save_path = os.path.join(save_path, f'{project}_best_model.pth')
            torch.save(best_model["model_state_dict"], best_model_save_path)
            # print(f'Saved best model to {best_model_save_path}')

            print(f'New best model found with F1 Score: {best_f1_score:.4f} at epoch {epoch + 1}')

            # Save the best model info as JSON
            best_model_info = {
                'epoch': best_model['epoch'],
                'f1_score': best_model['f1_score'],
                'model_save_path': best_model_save_path
            }
            best_model_json_path = os.path.join(save_path, f'{project}_best_model_info.json')
            with open(best_model_json_path, 'w') as json_file:
                json.dump(best_model_info, json_file)

        # Save the model after each epoch
        if only_save_best_model == False:
            model_save_path = os.path.join(save_path, f'{project}_model_epoch_{epoch + 1}_acc_{np.round(val_accuracy,4)}_f1_{np.round(val_f1,4)}.pth')
            torch.save(model.state_dict(), model_save_path)
        #print(f'Saved model to {model_save_path}')




        #print(f'Saved best model info to {best_model_json_path}')

        # Zeitmessung für die Epoche stoppen
        epoch_time = time.time() - epoch_start_time
        #print(f"Time taken for epoch {epoch + 1}: {epoch_time:.2f} seconds")

        # Speichern der Zeit für die Epoche in einer Datei
        with open(os.path.join(save_path, f'{project}_epoch_times.txt'), 'a') as f:
            f.write(f"Epoch {epoch + 1}: {epoch_time:.2f} seconds\n")

        #plotten







        # Gesamtzeit des Trainings messen
    total_training_time = time.time() - total_start_time
    print(f"Total training time: {total_training_time:.2f} seconds")

    # Speichern der Gesamtzeit in einer Datei
    with open(os.path.join(save_path, f'{project}_total_training_time.txt'), 'w') as f:
        f.write(f"Total training time: {total_training_time:.2f} seconds\n")

    print("Start Test Loop")
    # Test Loop
    model.eval()
    y_true = []
    y_pred = []
    # Initialisiere eine Liste, um die globalen Indizes zu speichern
    global_test_indices = []
    wrong_indices_by_fold = {}
    with torch.no_grad():
        for batch_idx, (esf_diff, extra_features, labels) in enumerate(test_loader):
            # Move inputs and labels to GPU (if available)
            esf_diff = esf_diff.to(device)
            labels = labels.to(device)

            outputs = model(esf_diff).unsqueeze(1)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

            if cv_info != None:
                wrong_predictions = predicted != labels
                # Extrahiere die Batch-individuellen Indizes
                batch_wrong_indices = torch.nonzero(wrong_predictions).squeeze()

                if batch_wrong_indices.numel() == 0:  # Prüfen, ob die Liste leer ist
                    print(f"Alle Vorhersagen im Batch {batch_idx} sind korrekt.")
                    continue  # Überspringe diesen Batch

                # Berechne die globalen Indizes innerhalb des Testloaders
                global_indices = batch_idx * test_loader.batch_size + batch_wrong_indices.cpu().numpy()

                if isinstance(global_indices, np.int64):
                    global_test_indices.extend([global_indices])
                    #print(batch_idx, global_test_indices)
                else:
                    global_test_indices.extend(global_indices)
                    #print(batch_idx, global_test_indices)


        global_test_indices  =  list(global_test_indices)




        print("wrong data labeled: ", len(global_test_indices))

        # Metriken berechnen
        test_accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average=average_metric)
        recall = recall_score(y_true, y_pred, average=average_metric)
        f1 = f1_score(y_true, y_pred, average=average_metric)

        # F1 scores for each label
        class_f1_scores = f1_score(y_true, y_pred, average=None)
        # Store F1 scores in a dictionary
        f1_scores_per_class = {f"Class {i}": f1 for i, f1 in enumerate(class_f1_scores)}

        # Example: Save to a file or log
        print("Class-wise F1 scores:", f1_scores_per_class)

        if cv_info != None:
            # Extrahiere die ursprünglichen Test-Indizes aus cv_info
            original_test_indices = cv_info[fold]["test"]

            # Zuordnung zu ursprünglichen Indizes im Datensatz
            original_wrong_indices = [original_test_indices[idx] for idx in global_test_indices]
            wrong_indices_by_fold[fold] = original_wrong_indices
            file_path = "wrong_labeled_data.json"  # Specify the file name or path
            with open(file_path, "a") as json_file:
                json.dump(wrong_indices_by_fold, json_file, indent=4)  # indent=4 for pretty formatting

        # Confusion Matrix
        conf_matrix = confusion_matrix(y_true, y_pred)
        TP = np.diag(conf_matrix)  # True Positives für jede Klasse
        FP = conf_matrix.sum(axis=0) - TP  # False Positives für jede Klasse
        FN = conf_matrix.sum(axis=1) - TP  # False Negatives für jede Klasse
        TN = conf_matrix.sum() - (FP + FN + TP)  # True Negatives für jede Klasse

        # Convert the confusion matrix to a DataFrame
        conf_matrix_df = pd.DataFrame(
            conf_matrix,
            index=[f"True_{i}" for i in range(conf_matrix.shape[0])],  # Rows: True labels
            columns=[f"Pred_{i}" for i in range(conf_matrix.shape[1])]  # Columns: Predicted labels
        )

        # Ergebnisse ausgeben
        print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1-Score: {f1:.4f}')
        print('Confusion Matrix:')
        print(conf_matrix)

        print(f'True Positives (TP): {TP}')
        print(f'False Positives (FP): {FP}')
        print(f'False Negatives (FN): {FN}')
        print(f'True Negatives (TN): {TN}')

        # Metriken berechnen für jede Klasse
        precision_per_class = TP / (TP + FP)
        recall_per_class = TP / (TP + FN)
        f1_per_class = 2 * (precision_per_class * recall_per_class) / (precision_per_class + recall_per_class)
        accuracy_per_class = (TP + TN) / (TP + TN + FP + FN)

        # Gesamtmetriken berechnen (gewichtete Durchschnittswerte über alle Klassen)
        total_TP = TP.sum()
        total_FP = FP.sum()
        total_FN = FN.sum()
        total_TN = TN.sum()

        total_precision = total_TP / (total_TP + total_FP)
        total_recall = total_TP / (total_TP + total_FN)
        total_f1 = 2 * (total_precision * total_recall) / (total_precision + total_recall)
        total_accuracy = (total_TP + total_TN) / (total_TP + total_TN + total_FP + total_FN)

        # Ergebnisse in DataFrame speichern
        metrics = {
            'Class': np.arange(len(TP)),
            'True Positives': TP,
            'False Positives': FP,
            'False Negatives': FN,
            'True Negatives': TN,
            'Precision': precision_per_class,
            'Recall': recall_per_class,
            'F1-Score': f1_per_class,
            'Accuracy': accuracy_per_class
        }

        df_metrics = pd.DataFrame(metrics)

        # Gesamtmetriken hinzufügen
        total_metrics = {
            'Class': 'Total',
            'True Positives': total_TP,
            'False Positives': total_FP,
            'False Negatives': total_FN,
            'True Negatives': total_TN,
            'Precision': total_precision,
            'Recall': total_recall,
            'F1-Score': total_f1,
            'Accuracy': total_accuracy
        }

        df_total_metrics = pd.DataFrame(total_metrics, index=[0])
        df_metrics = pd.concat([df_metrics, df_total_metrics], ignore_index=True)

        # DataFrame in eine Excel-Datei speichern
        excel_path = os.path.join(save_path, f'{project}_test_metrics.xlsx')
        with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
            df_metrics.to_excel(writer, sheet_name='Metrics_Per_Class', index=False)
            conf_matrix_df.to_excel(writer, sheet_name='Confusion_Matrix')

        print(f'Saved test metrics to {excel_path}')


    return test_accuracy, precision, recall, f1


def train_and_evaluate(model, train_loader, val_loader, test_loader, epochs=50, learning_rate=0.001, device = "cpu", save_path='./models', project = "default", only_save_best_model = False, cv_info=None, fold = None):
    # Move the model to the GPU (if available)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    average_metric = "macro" # 'macro' , 'weighted

    # Focal Loss initialisieren
    #criterion = FocalLoss(alpha=0.25, gamma=2)

    #optimizer = optim.Adam(model.parameters(), lr=learning_rate)#,  weight_decay=0.0001)#, weight_decay=0.01) #optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9) #optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    #optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    scheduler = DualMetricScheduler(optimizer, patience=10, mode="max")

    #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=setup["learning_rate"], steps_per_epoch=len(train_loader),epochs=setup["epochs"], pct_start=0.1)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    # Ensure save directory exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # Definiere das Log-Verzeichnis
    logdir = os.path.join(save_path, 'logs')
    writer = tf.summary.create_file_writer(logdir)
    # tensorboard --logdir= logdir

    # Initialize lists to store history
    train_losses = []
    val_losses = []
    train_f1s = []
    val_f1s = []
    train_accuracies = []
    val_accuracies = []
    best_f1_score = 0
    best_model = None
    mini_batches = True
    accumulation_steps = 4  # Simuliert größere Batchgröße

    total_start_time = time.time()

    # Initialisiere die Subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    # Training Loop
    for epoch in range(epochs):
        epoch_start_time = time.time()  # Zeitmessung für die Epoche starten
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        all_labels_train = []
        all_predictions_train = []


        for i, (inputs, labels) in enumerate(train_loader):
            # Move inputs and labels to GPU (if available)
            inputs, labels = inputs.to(device), labels.to(device)
            # Vorwärtsdurchlauf
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            #loss = combined_loss_function(outputs, labels, alpha=0.5)

            if mini_batches == True:
                loss = loss / accumulation_steps  # Loss normalisieren für Accumulation

            # Rückwärtsdurchlauf
            loss.backward()

            if (i + 1) % accumulation_steps == 0 and mini_batches == True:
                optimizer.step()
                optimizer.zero_grad() # Gradienten zurücksetzen

            # Ohne Gradient Accumulation: Standard-Optimierung
            if not mini_batches:
                optimizer.step()
                optimizer.zero_grad()
                running_loss += loss.item()

            else:
                # Laufenden Verlust berechnen
                running_loss += loss.item() * accumulation_steps  # Rückskalieren



            # Compute training accuracy
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            # Collect all labels and predictions for F1 score calculation
            all_labels_train.extend(labels.cpu().numpy())
            all_predictions_train.extend(predicted.cpu().numpy())

        train_accuracy = correct_train / total_train
        train_f1 = f1_score(all_labels_train, all_predictions_train, average=average_metric)

        train_losses.append(f"{epoch + 1}; {running_loss / len(train_loader):.4f}")
        train_accuracies.append(f"{epoch + 1}; {train_accuracy:.4f}")
        train_f1s.append(f"{epoch + 1}; {train_f1:.4f}")

        # Update train history files
        with open(os.path.join(save_path, f'{project}_train_loss.txt'), 'a') as f:
            f.write(f"{epoch + 1}; {running_loss / len(train_loader):.4f}\n")

        with open(os.path.join(save_path, f'{project}_train_accuracy.txt'), 'a') as f:
            f.write(f"{epoch + 1}; {train_accuracy:.4f}\n")

        with open(os.path.join(save_path, f'{project}_train_f1.txt'), 'a') as f:
            f.write(f"{epoch + 1}; {train_f1:.4f}\n")

        # Log metrics to TensorBoard
        with writer.as_default():
            tf.summary.scalar('Loss/train', running_loss / len(train_loader), step=epoch)
            tf.summary.scalar('Accuracy/train', train_accuracy, step=epoch)
            tf.summary.scalar('F1/train', train_f1, step=epoch)
            writer.flush()  # Sicherstellen, dass die Daten geschrieben werden



        # Validation Loop
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                # Move inputs and labels to GPU (if available)
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                #loss = combined_loss_function(outputs, labels, alpha=0.5)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

                # Collect all labels and predictions for F1 score calculation
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

                # Scheduler Schritt nach der Epoche


        val_accuracy = correct_val / total_val

        #val_f1 = f1_score(all_labels, all_predictions, average='weighted')
        val_f1 = f1_score(all_labels, all_predictions, average=average_metric)
        # F1 scores for each label
        class_f1_scores = f1_score(all_labels, all_predictions, average=None)
        # Store F1 scores in a dictionary
        f1_scores_per_class = {f"Class {i}": f1 for i, f1 in enumerate(class_f1_scores)}

        # Example: Save to a file or log
        print("Class-wise F1 scores:", f1_scores_per_class)

        #scheduler.step(val_loss)
        scheduler.step(val_loss, val_f1)

        val_losses.append(f"{epoch + 1}; {val_loss / len(val_loader):.4f}")
        val_accuracies.append(f"{epoch + 1}; {val_accuracy:.4f}")
        val_f1s.append(f"{epoch + 1}; {val_f1:.4f}")



        # Update validation history files
        with open(os.path.join(save_path, f'{project}_val_loss.txt'), 'a') as f:
            f.write(f"{epoch + 1}; {val_loss / len(val_loader):.4f}\n")

        with open(os.path.join(save_path, f'{project}_val_accuracy.txt'), 'a') as f:
            f.write(f"{epoch + 1}; {val_accuracy:.4f}\n")

        with open(os.path.join(save_path, f'{project}_val_f1.txt'), 'a') as f:
            f.write(f"{epoch + 1}; {val_f1:.4f}\n")

        # Print metrics
        print(
            f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_losses[-1].split("; ")[1]}, Train Accuracy: {train_accuracy:.4f}, Train F1 Score: {train_f1:.4f}, Val Loss: {val_losses[-1].split("; ")[1]}, Val Accuracy: {val_accuracy:.4f}, Val F1 Score: {val_f1:.4f}')

        # Log validation metrics to TensorBoard
        with writer.as_default():
            tf.summary.scalar('Loss/val', val_loss / len(val_loader), step=epoch)
            tf.summary.scalar('Accuracy/val', val_accuracy, step=epoch)
            tf.summary.scalar('F1/val', val_f1, step=epoch)
            writer.flush()

        # Check if this is the best model based on F1 score
        if val_f1 > best_f1_score:
            best_f1_score = val_f1
            best_model = {
                'epoch': epoch + 1,
                'f1_score': best_f1_score,
                'model_state_dict': model.state_dict()
            }

            # Save the best model
            best_model_save_path = os.path.join(save_path, f'{project}_best_model.pth')
            torch.save(best_model["model_state_dict"], best_model_save_path)
            # print(f'Saved best model to {best_model_save_path}')

            print(f'New best model found with F1 Score: {best_f1_score:.4f} at epoch {epoch + 1}')

            # Save the best model info as JSON
            best_model_info = {
                'epoch': best_model['epoch'],
                'f1_score': best_model['f1_score'],
                'model_save_path': best_model_save_path
            }
            best_model_json_path = os.path.join(save_path, f'{project}_best_model_info.json')
            with open(best_model_json_path, 'w') as json_file:
                json.dump(best_model_info, json_file)

        # Save the model after each epoch
        if only_save_best_model == False:
            model_save_path = os.path.join(save_path, f'{project}_model_epoch_{epoch + 1}_acc_{np.round(val_accuracy,4)}_f1_{np.round(val_f1,4)}.pth')
            torch.save(model.state_dict(), model_save_path)
        #print(f'Saved model to {model_save_path}')




        #print(f'Saved best model info to {best_model_json_path}')

        # Zeitmessung für die Epoche stoppen
        epoch_time = time.time() - epoch_start_time
        #print(f"Time taken for epoch {epoch + 1}: {epoch_time:.2f} seconds")

        # Speichern der Zeit für die Epoche in einer Datei
        with open(os.path.join(save_path, f'{project}_epoch_times.txt'), 'a') as f:
            f.write(f"Epoch {epoch + 1}: {epoch_time:.2f} seconds\n")

        #plotten







        # Gesamtzeit des Trainings messen
    total_training_time = time.time() - total_start_time
    print(f"Total training time: {total_training_time:.2f} seconds")

    # Speichern der Gesamtzeit in einer Datei
    with open(os.path.join(save_path, f'{project}_total_training_time.txt'), 'w') as f:
        f.write(f"Total training time: {total_training_time:.2f} seconds\n")

    print("Start Test Loop")
    # Test Loop
    model.eval()
    y_true = []
    y_pred = []
    # Initialisiere eine Liste, um die globalen Indizes zu speichern
    global_test_indices = []
    wrong_indices_by_fold = {}
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            # Move inputs and labels to GPU (if available)
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

            if cv_info != None:
                wrong_predictions = predicted != labels
                # Extrahiere die Batch-individuellen Indizes
                batch_wrong_indices = torch.nonzero(wrong_predictions).squeeze()

                if batch_wrong_indices.numel() == 0:  # Prüfen, ob die Liste leer ist
                    print(f"Alle Vorhersagen im Batch {batch_idx} sind korrekt.")
                    continue  # Überspringe diesen Batch

                # Berechne die globalen Indizes innerhalb des Testloaders
                global_indices = batch_idx * test_loader.batch_size + batch_wrong_indices.cpu().numpy()

                if isinstance(global_indices, np.int64):
                    global_test_indices.extend([global_indices])
                    #print(batch_idx, global_test_indices)
                else:
                    global_test_indices.extend(global_indices)
                    #print(batch_idx, global_test_indices)


        global_test_indices  =  list(global_test_indices)




        print("wrong data labeled: ", len(global_test_indices))

        # Metriken berechnen
        test_accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average=average_metric)
        recall = recall_score(y_true, y_pred, average=average_metric)
        f1 = f1_score(y_true, y_pred, average=average_metric)

        # F1 scores for each label
        class_f1_scores = f1_score(y_true, y_pred, average=None)
        # Store F1 scores in a dictionary
        f1_scores_per_class = {f"Class {i}": f1 for i, f1 in enumerate(class_f1_scores)}

        # Example: Save to a file or log
        print("Class-wise F1 scores:", f1_scores_per_class)

        if cv_info != None:
            # Extrahiere die ursprünglichen Test-Indizes aus cv_info
            original_test_indices = cv_info[fold]["test"]

            # Zuordnung zu ursprünglichen Indizes im Datensatz
            original_wrong_indices = [original_test_indices[idx] for idx in global_test_indices]
            wrong_indices_by_fold[fold] = original_wrong_indices
            file_path = "wrong_labeled_data.json"  # Specify the file name or path
            with open(file_path, "a") as json_file:
                json.dump(wrong_indices_by_fold, json_file, indent=4)  # indent=4 for pretty formatting

        # Confusion Matrix
        conf_matrix = confusion_matrix(y_true, y_pred)
        TP = np.diag(conf_matrix)  # True Positives für jede Klasse
        FP = conf_matrix.sum(axis=0) - TP  # False Positives für jede Klasse
        FN = conf_matrix.sum(axis=1) - TP  # False Negatives für jede Klasse
        TN = conf_matrix.sum() - (FP + FN + TP)  # True Negatives für jede Klasse

        # Convert the confusion matrix to a DataFrame
        conf_matrix_df = pd.DataFrame(
            conf_matrix,
            index=[f"True_{i}" for i in range(conf_matrix.shape[0])],  # Rows: True labels
            columns=[f"Pred_{i}" for i in range(conf_matrix.shape[1])]  # Columns: Predicted labels
        )

        # Ergebnisse ausgeben
        print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1-Score: {f1:.4f}')
        print('Confusion Matrix:')
        print(conf_matrix)

        print(f'True Positives (TP): {TP}')
        print(f'False Positives (FP): {FP}')
        print(f'False Negatives (FN): {FN}')
        print(f'True Negatives (TN): {TN}')

        # Metriken berechnen für jede Klasse
        precision_per_class = TP / (TP + FP)
        recall_per_class = TP / (TP + FN)
        f1_per_class = 2 * (precision_per_class * recall_per_class) / (precision_per_class + recall_per_class)
        accuracy_per_class = (TP + TN) / (TP + TN + FP + FN)

        # Gesamtmetriken berechnen (gewichtete Durchschnittswerte über alle Klassen)
        total_TP = TP.sum()
        total_FP = FP.sum()
        total_FN = FN.sum()
        total_TN = TN.sum()

        total_precision = total_TP / (total_TP + total_FP)
        total_recall = total_TP / (total_TP + total_FN)
        total_f1 = 2 * (total_precision * total_recall) / (total_precision + total_recall)
        total_accuracy = (total_TP + total_TN) / (total_TP + total_TN + total_FP + total_FN)

        # Ergebnisse in DataFrame speichern
        metrics = {
            'Class': np.arange(len(TP)),
            'True Positives': TP,
            'False Positives': FP,
            'False Negatives': FN,
            'True Negatives': TN,
            'Precision': precision_per_class,
            'Recall': recall_per_class,
            'F1-Score': f1_per_class,
            'Accuracy': accuracy_per_class
        }

        df_metrics = pd.DataFrame(metrics)

        # Gesamtmetriken hinzufügen
        total_metrics = {
            'Class': 'Total',
            'True Positives': total_TP,
            'False Positives': total_FP,
            'False Negatives': total_FN,
            'True Negatives': total_TN,
            'Precision': total_precision,
            'Recall': total_recall,
            'F1-Score': total_f1,
            'Accuracy': total_accuracy
        }

        df_total_metrics = pd.DataFrame(total_metrics, index=[0])
        df_metrics = pd.concat([df_metrics, df_total_metrics], ignore_index=True)

        # DataFrame in eine Excel-Datei speichern
        excel_path = os.path.join(save_path, f'{project}_test_metrics.xlsx')
        with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
            df_metrics.to_excel(writer, sheet_name='Metrics_Per_Class', index=False)
            conf_matrix_df.to_excel(writer, sheet_name='Confusion_Matrix')

        print(f'Saved test metrics to {excel_path}')


    return test_accuracy, precision, recall, f1


import os, time, json, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch import optim
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, accuracy_score

# =======================
# Hilfsfunktionen
# =======================

def _bce_weighted(logits, targets, w_neg=1.0, w_pos=1.0):
    """
    logits: (B,1)
    targets: (B,1) float {0,1}
    Elementweise Gewichte:
      - Label 0 -> w_neg
      - Label 1 -> w_pos
    """
    targets = targets.float()
    per_el = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    w = torch.where(targets > 0.5,
                    torch.as_tensor(w_pos, device=logits.device),
                    torch.as_tensor(w_neg, device=logits.device))
    return (per_el * w).mean()

def _search_best_f1(y_true, y_prob, thresholds=None, topk_print=5, prefix="VAL"):
    """
    Iterative Schwellen-Suche für bestes F1.
    Debug: Top-K mit Prints.
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    best = {"thr":0.5,"f1":-1,"prec":0.0,"rec":0.0,"cov":0.0}
    rows = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f = f1_score(y_true, y_pred, zero_division=0)
        c = (y_pred == 1).mean()
        rows.append((t, f, p, r, c))
        if f > best["f1"]:
            best = {"thr":float(t), "f1":float(f), "prec":float(p), "rec":float(r), "cov":float(c)}

    print(f"[{prefix}] Top-{topk_print} thresholds by F1:")
    for t, f, p, r, c in sorted(rows, key=lambda r: r[1], reverse=True)[:topk_print]:
        print(f"  thr={t:.3f} | F1={f:.4f} | P={p:.4f} | R={r:.4f} | cov_pos={c:.3f}")

    print(f"[{prefix}] Best@F1: t*={best['thr']:.3f} | F1={best['f1']:.4f} | "
          f"P={best['prec']:.4f} | R={best['rec']:.4f} | cov={best['cov']:.3f}")
    return best, rows

def _pick_for_precision(y_true, y_prob, precision_target=0.95, thresholds=None, prefix="VAL"):
    """
    Kleinstes t mit Precision >= Ziel.
    Fallback t=0.999.
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.999, 200)

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        p = precision_score(y_true, y_pred, zero_division=0)
        if p >= precision_target:
            r = recall_score(y_true, y_pred, zero_division=0)
            f = f1_score(y_true, y_pred, zero_division=0)
            c = (y_pred == 1).mean()
            print(f"[{prefix}] Pick@P≥{precision_target:.2f}: thr={t:.3f} | P={p:.4f} | R={r:.4f} | F1={f:.4f} | cov={c:.3f}")
            return {"thr":float(t), "prec":float(p), "rec":float(r), "f1":float(f), "cov":float(c)}

    # Fallback
    t = 0.999
    y_pred = (y_prob >= t).astype(int)
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f = f1_score(y_true, y_pred, zero_division=0)
    c = (y_pred == 1).mean()
    print(f"[{prefix}] Ziel-Precision {precision_target:.2f} nicht erreichbar. Fallback thr=0.999 | "
          f"P={p:.4f} | R={r:.4f} | F1={f:.4f} | cov={c:.3f}")
    return {"thr":float(t), "prec":float(p), "rec":float(r), "f1":float(f), "cov":float(c)}

# ---------- Temperature Scaling (optional) ----------

class _TempScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_t = nn.Parameter(torch.zeros(1))  # T = softplus(log_t)+eps

    def forward(self, logits):
        T = F.softplus(self.log_t) + 1e-6
        return logits / T

def _fit_temperature_on_val(logits_np, labels_np, steps=250, lr=0.05, verbose=True):
    """
    logits_np: (N,1) rohen Logits vom Val-Set.
    labels_np: (N,1) float {0,1}.
    Minimiert BCE-NLL durch T-Skalierung.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _TempScaler().to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    logits = torch.from_numpy(logits_np).float().to(device)
    labels = torch.from_numpy(labels_np).float().to(device)

    for s in range(steps):
        opt.zero_grad()
        logits_t = model(logits)
        loss = F.binary_cross_entropy_with_logits(logits_t, labels)
        loss.backward()
        opt.step()
        if verbose and (s % 50 == 0 or s == steps - 1):
            with torch.no_grad():
                T = float(F.softplus(model.log_t) + 1e-6)
            print(f"[CAL] step={s:03d} | NLL={loss.item():.5f} | T={T:.4f}")

    with torch.no_grad():
        T = float(F.softplus(model.log_t) + 1e-6)
    return T


def _to_device(dev):
    import torch
    return torch.device(dev) if isinstance(dev, str) else dev

def _is_cuda(dev):
    return getattr(dev, "type", None) == "cuda"

# =======================
# Hauptfunktion
# =======================



def train_and_evaluate_scan2bim(
    model,
    train_loader, val_loader, test_loader,
    epochs=50, learning_rate=1e-3, device="cpu",
    save_path="./models", project="default",
    accumulation_steps=4, use_amp=True,
    # Kostenprofil: FP(1) teuer -> negatives höher gewichten
    neg_weight=4.0, pos_weight=1.0,
    # Schwellen-Suche
    thr_grid=np.linspace(0.01, 0.99, 99),
    precision_target_pos=0.95,    # konservativ, kann None sein
    # Temperature Scaling
    calibrate_temperature=True,
):
    """
    Erwartet: model(esf_diff, extra_features) -> (B,1) Logits.
    Labels: float {0,1}.
    Speichert:
      - bestes Modell nach F1@t* (und zeigt Precision-Pick)
      - JSON mit t_best_f1, t_prec_target (falls gesetzt), Temperatur T
      - Excel-Report für Test
    """
    # robustes Device-Setup
    device = _to_device(device)
    on_cuda = _is_cuda(device)

    amp_enabled = bool(use_amp) and on_cuda
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    model = model.to(device)
    os.makedirs(save_path, exist_ok=True)
    best_info_path = os.path.join(save_path, f"{project}_best_model_info.json")
    best_model_path = os.path.join(save_path, f"{project}_best_model.pth")

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)


    best_tuple = (-1.0, -1.0, -1.0)  # (F1, Precision, Recall) @ t*
    best_val_dump = {}
    total_start = time.time()

    # =======================
    # Training
    # =======================
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        ep_start = time.time()

        running_loss = 0.0
        n_batches = len(train_loader)

        # für Info @0.5 (optional)
        tr_prob, tr_true = [], []

        for i, (esf_diff, extra_features, labels) in enumerate(train_loader):
            esf_diff = esf_diff.to(device)
            extra_features = extra_features.to(device)
            labels = labels.to(device).float().view(-1, 1)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits = model(esf_diff, extra_features)
                loss = _bce_weighted(logits, labels, w_neg=neg_weight, w_pos=pos_weight)

            if accumulation_steps and accumulation_steps > 1:
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            last_batch = (i == n_batches - 1)
            step_cond = (accumulation_steps and accumulation_steps > 1 and ((i + 1) % accumulation_steps == 0 or last_batch)) \
                        or (not accumulation_steps or accumulation_steps == 1)

            if step_cond:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # Logs
            with torch.no_grad():
                if accumulation_steps and accumulation_steps > 1:
                    running_loss += float(loss.item() * accumulation_steps)
                else:
                    running_loss += float(loss.item())
                tr_prob.append(torch.sigmoid(logits).detach().cpu().numpy().ravel())
                tr_true.append(labels.detach().cpu().numpy().ravel())

        # Train-Info @0.5
        tr_prob = np.concatenate(tr_prob, axis=0)
        tr_true = np.concatenate(tr_true, axis=0).astype(int)
        y_tr_05 = (tr_prob >= 0.5).astype(int)
        tr_acc_05 = accuracy_score(tr_true, y_tr_05)
        tr_f1_05 = f1_score(tr_true, y_tr_05, zero_division=0)

        # =======================
        # Validation
        # =======================
        model.eval()
        val_loss_sum = 0.0
        val_prob, val_true = [], []
        val_logits_all = []

        with torch.no_grad():
            for (esf_diff, extra_features, labels) in val_loader:
                esf_diff = esf_diff.to(device)
                extra_features = extra_features.to(device)
                labels = labels.to(device).float().view(-1, 1)
                logits = model(esf_diff, extra_features)
                loss = _bce_weighted(logits, labels, w_neg=neg_weight, w_pos=pos_weight)
                val_loss_sum += float(loss.item())

                p1 = torch.sigmoid(logits).cpu().numpy().ravel()
                val_prob.append(p1)
                val_true.append(labels.cpu().numpy().ravel())
                val_logits_all.append(logits.cpu().numpy().ravel())

        val_prob = np.concatenate(val_prob, axis=0)
        val_true = np.concatenate(val_true, axis=0).astype(int)
        val_logits_all = np.concatenate(val_logits_all, axis=0).reshape(-1, 1)

        best_f1_pick, _ = _search_best_f1(val_true, val_prob, thresholds=thr_grid, topk_print=5, prefix="VAL")
        t_best = best_f1_pick["thr"]

        # optional konservativer Pick
        prec_pick = None
        if precision_target_pos is not None:
            prec_pick = _pick_for_precision(val_true, val_prob, precision_target=precision_target_pos,
                                            thresholds=np.linspace(0.01, 0.999, 200), prefix="VAL")

        # Metriken @ t_best
        y_val_best = (val_prob >= t_best).astype(int)
        val_f1 = f1_score(val_true, y_val_best, zero_division=0)
        val_p = precision_score(val_true, y_val_best, zero_division=0)
        val_r = recall_score(val_true, y_val_best, zero_division=0)
        val_a = accuracy_score(val_true, y_val_best)
        val_loss = val_loss_sum / max(len(val_loader), 1)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {running_loss/len(train_loader):.4f} "
              f"| Train Acc@0.5: {tr_acc_05:.4f} | Train F1@0.5: {tr_f1_05:.4f}")
        print(f"Epoch {epoch+1}/{epochs} | Val Loss: {val_loss:.4f} | Val Acc@t*: {val_a:.4f} "
              f"| Val F1@t*: {val_f1:.4f} | P={val_p:.4f} | R={val_r:.4f} | t*={t_best:.3f}")
        print(f"Epoch time: {time.time()-ep_start:.2f}s")

        # Auswahl: bestes Modell nach F1@t*
        cur_tuple = (val_f1, val_p, val_r)
        if cur_tuple > best_tuple:
            best_tuple = cur_tuple
            torch.save(model.state_dict(), best_model_path)
            best_val_dump = {
                "epoch": epoch+1,
                "val_loss": val_loss,
                "t_best_f1": float(t_best),
                "val_f1": float(val_f1),
                "val_precision": float(val_p),
                "val_recall": float(val_r),
                "precision_target_pick": prec_pick,
            }
            with open(best_info_path, "w") as f:
                json.dump(best_val_dump, f, indent=2)
            print(f"New best model saved @ epoch {epoch+1} | F1={val_f1:.4f} | t*={t_best:.3f}")

    # =======================
    # Temperature Scaling (optional)
    # =======================
    T = 1.0
    if calibrate_temperature:
        # lade bestes Modell
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.eval()
        all_logits, all_labels = [], []
        with torch.no_grad():
            for (esf_diff, extra_features, labels) in val_loader:
                logits = model(esf_diff.to(device), extra_features.to(device))
                all_logits.append(logits.cpu().numpy().ravel())
                all_labels.append(labels.numpy().ravel())
        all_logits = np.concatenate(all_logits, 0).reshape(-1,1)
        all_labels = np.concatenate(all_labels, 0).reshape(-1,1).astype(np.float32)
        T = _fit_temperature_on_val(all_logits, all_labels, steps=250, lr=0.05, verbose=True)

        # Update JSON
        best_val_dump["temperature"] = float(T)
        with open(best_info_path, "w") as f:
            json.dump(best_val_dump, f, indent=2)

    # =======================
    # Test
    # =======================
    print("Start Test Loop")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Lese t_best
    t_best = best_val_dump.get("t_best_f1", 0.5)
    # Wenn konservativ: nimm Precision-Pick
    if precision_target_pos is not None and isinstance(best_val_dump.get("precision_target_pick"), dict):
        t_best = best_val_dump["precision_target_pick"]["thr"]

    y_true, y_prob = [], []
    with torch.no_grad():
        for (esf_diff, extra_features, labels) in test_loader:
            logits = model(esf_diff.to(device), extra_features.to(device))  # (B,1)
            # Temperature Scaling bei Inferenz
            logits = logits / T
            p1 = torch.sigmoid(logits).cpu().numpy().ravel()
            y_prob.append(p1)
            y_true.append(labels.numpy().ravel())

    y_prob = np.concatenate(y_prob, 0)
    y_true = np.concatenate(y_true, 0).astype(int)
    y_pred = (y_prob >= t_best).astype(int)

    test_acc = accuracy_score(y_true, y_pred)
    test_prec = precision_score(y_true, y_pred, zero_division=0)
    test_rec = recall_score(y_true, y_pred, zero_division=0)
    test_f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f"Test @ t={t_best:.3f}, T={T:.3f} | Acc={test_acc:.4f} | P={test_prec:.4f} | R={test_rec:.4f} | F1={test_f1:.4f}")

    # Confusion + per-class
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    TP = np.diag(cm)
    FP = cm.sum(axis=0) - TP
    FN = cm.sum(axis=1) - TP
    TN = cm.sum() - (FP + FN + TP)

    print("Confusion Matrix:\n", cm)
    print(f"TP={TP}, FP={FP}, FN={FN}, TN={TN}")

    precision_per_class = TP / np.clip(TP + FP, 1, None)
    recall_per_class    = TP / np.clip(TP + FN, 1, None)
    f1_per_class        = 2 * (precision_per_class * recall_per_class) / np.clip(precision_per_class + recall_per_class, 1e-9, None)
    acc_per_class       = (TP + TN) / np.clip(TP + TN + FP + FN, 1, None)

    # Business-KPI
    pos_preds = (y_pred == 1).sum()
    fp1_per_100 = 100.0 * (FP[1] / max(pos_preds, 1))
    print(f"FP(1) pro 100 gemeldete 'vorhanden': {fp1_per_100:.2f}")

    # Excel
    conf_df = pd.DataFrame(cm, index=["True_0","True_1"], columns=["Pred_0","Pred_1"])
    df_metrics = pd.DataFrame({
        "Class":[0,1],
        "TP":TP, "FP":FP, "FN":FN, "TN":TN,
        "Precision":precision_per_class, "Recall":recall_per_class,
        "F1-Score":f1_per_class, "Accuracy":acc_per_class
    })
    df_total = pd.DataFrame({
        "Class":["Total"],
        "TP":[TP.sum()], "FP":[FP.sum()], "FN":[FN.sum()], "TN":[TN.sum()],
        "Precision":[test_prec], "Recall":[test_rec], "F1-Score":[test_f1], "Accuracy":[test_acc]
    })
    df_all = pd.concat([df_metrics, df_total], ignore_index=True)

    excel_path = os.path.join(save_path, f"{project}_test_metrics.xlsx")
    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        df_all.to_excel(writer, sheet_name="Metrics_Per_Class", index=False)
        conf_df.to_excel(writer, sheet_name="Confusion_Matrix")

    print(f"Saved test metrics to {excel_path}")
    print(f"Total training time: {time.time()-total_start:.2f}s")
    return test_acc, test_prec, test_rec, test_f1, t_best, T