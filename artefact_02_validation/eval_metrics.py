import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sensitivity = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall_class0 = recall_score(y_true, y_pred, pos_label=0, zero_division=0)

    metrics = {
        "samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "recall": float(sensitivity),
        "sensitivity": float(sensitivity),
        "specificity": float(recall_class0),
        "fp1_rate": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        "fp0_rate": float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }
    return metrics


def _save_confusion_matrix(cm: np.ndarray, title: str, out_file: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["pred_0", "pred_1"])
    ax.set_yticklabels(["gt_0", "gt_1"])
    ax.set_title(title)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

    fig.tight_layout()
    fig.savefig(out_file, dpi=200)
    plt.close(fig)


def _evaluate_subset(df: pd.DataFrame, out_dir: Path, prefix: str) -> Dict[str, float]:
    eval_df = df.dropna(subset=["ground_truth"]).copy()
    if eval_df.empty:
        return {}

    invalid_mask = ~eval_df["ground_truth"].astype(int).isin([0, 1])
    if invalid_mask.any():
        invalid_values = sorted(eval_df.loc[invalid_mask, "ground_truth"].astype(int).unique().tolist())
        raise ValueError(f"Only ground_truth labels 0/1 are allowed. Found: {invalid_values}")

    y_true = eval_df["ground_truth"].astype(int).to_numpy()
    y_pred = eval_df["predicted_class"].astype(int).to_numpy()

    metrics = _compute_metrics(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    _save_confusion_matrix(cm, f"{prefix} confusion matrix", out_dir / f"{prefix}_confusion_matrix.png")

    meta_cols = ["checkpoint_path", "checkpoint_sha256", "source_file"]
    meta_snapshot = eval_df[meta_cols].drop_duplicates().to_dict(orient="records")
    metrics["meta_records"] = len(meta_snapshot)

    with (out_dir / f"{prefix}_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta_snapshot, f, indent=2, ensure_ascii=False)

    return metrics


def evaluate(input_file: Path) -> Path:
    df = pd.read_excel(input_file) if input_file.suffix.lower() in {".xlsx", ".xls"} else pd.read_json(input_file)
    out_dir = input_file.parent / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = []

    for project, df_project in df.groupby("project"):
        m = _evaluate_subset(df_project, out_dir, f"project_{project}")
        if m:
            m["scope"] = "project"
            m["name"] = project
            all_metrics.append(m)

    for domain, df_domain in df.groupby("domain"):
        m = _evaluate_subset(df_domain, out_dir, f"domain_{domain}")
        if m:
            m["scope"] = "domain"
            m["name"] = domain
            all_metrics.append(m)

    global_metrics = _evaluate_subset(df, out_dir, "all_projects")
    if global_metrics:
        global_metrics["scope"] = "global"
        global_metrics["name"] = "all_projects"
        all_metrics.append(global_metrics)

    if not all_metrics:
        raise RuntimeError("No rows with ground_truth found. Please label data first.")

    metrics_df = pd.DataFrame(all_metrics)
    metrics_file = out_dir / "metrics_summary.xlsx"
    metrics_df.to_excel(metrics_file, index=False)

    with (out_dir / "metrics_summary.json").open("w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    print(f"Saved metrics to: {metrics_file}")
    return metrics_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate validation predictions with ground-truth labels.")
    parser.add_argument("--input", required=True, help="Path to prediction file (xlsx or json).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(Path(args.input))
