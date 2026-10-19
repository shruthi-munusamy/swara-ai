"""
Evaluate the trained LSTM raga identifier on the held-out test set.

Prerequisites:
    python scripts/prepare_identifier_data.py
    python scripts/train_identifier.py

Usage:
    python scripts/evaluate_identifier.py

Outputs:
    models/confusion_matrix.png
    models/confusion_matrix.csv
    models/test_predictions.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.identifier.dataset import NUM_SWARA_CLASSES, discover_tracks
from src.identifier.evaluate import allied_raga_overlap, evaluate_predictions, predict_tracks, top_confusion_pairs
from src.identifier.model import RagaLSTM

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
DEFAULT_SARAGA_ROOT = Path("data/raw/saraga/saraga1.5_carnatic")
DEFAULT_PROCESSED_DIR = Path("data/processed")


def _load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _plot_confusion_matrix(confusion, label_names, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(confusion, cmap="Blues")
    ax.set_xticks(range(len(label_names)))
    ax.set_yticks(range(len(label_names)))
    ax.set_xticklabels(label_names, rotation=45, ha="right")
    ax.set_yticklabels(label_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Raga identifier confusion matrix (test set)")
    threshold = confusion.max() / 2 if confusion.max() > 0 else 0
    for i in range(len(label_names)):
        for j in range(len(label_names)):
            val = confusion[i, j]
            if val > 0:
                ax.text(j, i, str(val), ha="center", va="center", color="white" if val > threshold else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--saraga-root", type=Path, default=DEFAULT_SARAGA_ROOT)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()

    config = _load_config(args.config)
    id_cfg = config["identifier"]

    with open(args.processed_dir / "mvp_ragas.yaml", "r", encoding="utf-8") as f:
        mvp = yaml.safe_load(f)
    with open(args.processed_dir / "splits.yaml", "r", encoding="utf-8") as f:
        splits = yaml.safe_load(f)

    raga_to_label = {name: i for i, name in enumerate(mvp["ragas"])}
    label_names = mvp["ragas"]

    tracks_by_id = {t.track_id: t for t in discover_tracks(args.saraga_root)}
    test_tracks = [tracks_by_id[tid] for tid in splits["test"] if tid in tracks_by_id]
    missing = [tid for tid in splits["test"] if tid not in tracks_by_id]
    if missing:
        print(f"warning: {len(missing)} test track_id(s) no longer resolve, skipping", file=sys.stderr)

    checkpoint_path = args.checkpoint or Path(config["paths"]["models_dir"]) / "identifier_best.pt"
    model = RagaLSTM(
        num_swara_classes=NUM_SWARA_CLASSES,
        num_ragas=len(raga_to_label),
        embedding_dim=id_cfg["embedding_dim"],
        hidden_size=id_cfg["hidden_size"],
        num_layers=id_cfg["num_layers"],
        dropout=id_cfg["dropout"],
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

    pipeline_kwargs = dict(
        sample_rate=config["audio"]["sample_rate"],
        crepe_model=config["pitch"]["crepe_model"],
        step_size_ms=config["pitch"]["step_size_ms"],
        confidence_threshold=config["pitch"]["confidence_threshold"],
        tolerance_cents=config["swara"]["tolerance_cents"],
        min_stable_ms=config["swara"]["min_stable_ms"],
        smoothing_window=config["swara"]["smoothing_window"],
    )
    cache_dir = Path(config["paths"]["data_dir"]) / "processed" / "features"

    predictions = predict_tracks(
        model, test_tracks, raga_to_label, cache_dir, pipeline_kwargs,
        id_cfg["window_size_segments"], id_cfg["window_stride_segments"], torch.device("cpu"),
    )
    report = evaluate_predictions(predictions, label_names)

    print(f"\nTest tracks evaluated: {len(predictions)}")
    print(f"Accuracy:     {report.accuracy:.3f}")
    print(f"Macro-F1:     {report.macro_f1:.3f}")
    print(f"Weighted-F1:  {report.weighted_f1:.3f}\n")
    print(f"{'Raga':20s} {'Precision':>10s} {'Recall':>10s} {'F1':>8s}")
    for name in label_names:
        print(
            f"{name:20s} {report.per_raga_precision[name]:10.3f} "
            f"{report.per_raga_recall[name]:10.3f} {report.per_raga_f1[name]:8.3f}"
        )

    models_dir = Path(config["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    cm_csv_path = models_dir / "confusion_matrix.csv"
    with open(cm_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([""] + label_names)
        for i, name in enumerate(label_names):
            writer.writerow([name] + report.confusion[i].tolist())
    print(f"\nWrote {cm_csv_path}")

    cm_png_path = models_dir / "confusion_matrix.png"
    _plot_confusion_matrix(report.confusion, label_names, cm_png_path)
    print(f"Wrote {cm_png_path}")

    pred_csv_path = models_dir / "test_predictions.csv"
    with open(pred_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["track_id", "true_raga", "predicted_raga", "confidence", "top3"])
        for p in predictions:
            top3_str = "; ".join(
                f"{label_names[i]}:{c:.3f}" for i, c in zip(p.top3_labels, p.top3_confidences)
            )
            writer.writerow(
                [p.track_id, label_names[p.true_label], label_names[p.predicted_label], f"{p.confidence:.3f}", top3_str]
            )
    print(f"Wrote {pred_csv_path}")

    top_pairs = top_confusion_pairs(report, top_k=5)
    if top_pairs:
        print("\nTop confusion pairs (true -> predicted, count, swara overlap):")
        for true_name, pred_name, count in top_pairs:
            verdict = allied_raga_overlap(true_name, pred_name)
            print(f"  {true_name} -> {pred_name}: {count}  ({verdict})")
    else:
        print("\nNo off-diagonal confusions.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
