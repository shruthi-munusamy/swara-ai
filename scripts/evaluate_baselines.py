"""
Train and evaluate the SVM/MLP baselines, and produce a comparison table
against the LSTM identifier.

Prerequisites:
    python scripts/prepare_identifier_data.py
    python scripts/train_identifier.py

Usage:
    python scripts/evaluate_baselines.py

All macro-F1 scores are track-level. The LSTM's are computed with the same
code path as scripts/evaluate_identifier.py.

Outputs:
    models/baseline_comparison.md
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.identifier.baselines import train_mlp_baseline, train_svm_baseline
from src.identifier.dataset import NUM_SWARA_CLASSES, discover_tracks
from src.identifier.evaluate import evaluate_predictions, predict_tracks
from src.identifier.model import RagaLSTM

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
DEFAULT_SARAGA_ROOT = Path("data/raw/saraga/saraga1.5_carnatic")
DEFAULT_PROCESSED_DIR = Path("data/processed")


def _load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _lstm_row(
    config: dict,
    raga_to_label: dict,
    val_tracks: list,
    test_tracks: list,
    cache_dir: Path,
    pipeline_kwargs: dict,
) -> dict:
    """
    Track-level val/test macro-F1, param count, and single-window inference
    latency for the trained LSTM checkpoint.
    """
    id_cfg = config["identifier"]
    model = RagaLSTM(
        num_swara_classes=NUM_SWARA_CLASSES,
        num_ragas=len(raga_to_label),
        embedding_dim=id_cfg["embedding_dim"],
        hidden_size=id_cfg["hidden_size"],
        num_layers=id_cfg["num_layers"],
        dropout=id_cfg["dropout"],
    )
    checkpoint_path = Path(config["paths"]["models_dir"]) / "identifier_best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"{checkpoint_path} not found. Run scripts/train_identifier.py first.")
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    label_names = sorted(raga_to_label, key=raga_to_label.get)

    def _macro_f1(tracks: list) -> float:
        predictions = predict_tracks(
            model, tracks, raga_to_label, cache_dir, pipeline_kwargs,
            id_cfg["window_size_segments"], id_cfg["window_stride_segments"], torch.device("cpu"),
        )
        return evaluate_predictions(predictions, label_names).macro_f1

    num_params = sum(p.numel() for p in model.parameters())

    dummy = torch.randint(0, NUM_SWARA_CLASSES, (1, id_cfg["window_size_segments"]))
    length = torch.tensor([id_cfg["window_size_segments"]])
    with torch.no_grad():
        start = time.perf_counter()
        for _ in range(20):
            model(dummy, length)
        elapsed = time.perf_counter() - start
    latency_ms = (elapsed / 20) * 1000.0

    return {
        "name": "BiLSTM+attention",
        "val_macro_f1": _macro_f1(val_tracks),
        "test_macro_f1": _macro_f1(test_tracks),
        "num_params": num_params,
        "mean_inference_latency_ms": latency_ms,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--saraga-root", type=Path, default=DEFAULT_SARAGA_ROOT)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    args = parser.parse_args()

    config = _load_config(args.config)
    id_cfg = config["identifier"]

    with open(args.processed_dir / "mvp_ragas.yaml", "r", encoding="utf-8") as f:
        mvp = yaml.safe_load(f)
    with open(args.processed_dir / "splits.yaml", "r", encoding="utf-8") as f:
        splits = yaml.safe_load(f)

    raga_to_label = {name: i for i, name in enumerate(mvp["ragas"])}
    tracks_by_id = {t.track_id: t for t in discover_tracks(args.saraga_root)}

    def _resolve(ids):
        return [tracks_by_id[tid] for tid in ids if tid in tracks_by_id]

    train_tracks = _resolve(splits["train"])
    val_tracks = _resolve(splits["val"])
    test_tracks = _resolve(splits["test"])

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
    window_size = id_cfg["window_size_segments"]
    window_stride = id_cfg["window_stride_segments"]

    print("Training SVM baseline...")
    svm_result = train_svm_baseline(
        train_tracks, val_tracks, test_tracks, raga_to_label, cache_dir, pipeline_kwargs, window_size, window_stride
    )
    print(f"  SVM: val_macro_f1={svm_result.val_macro_f1:.3f}  test_macro_f1={svm_result.test_macro_f1:.3f}")

    print("Training MLP baseline...")
    mlp_result = train_mlp_baseline(
        train_tracks, val_tracks, test_tracks, raga_to_label, cache_dir, pipeline_kwargs, window_size, window_stride
    )
    print(f"  MLP: val_macro_f1={mlp_result.val_macro_f1:.3f}  test_macro_f1={mlp_result.test_macro_f1:.3f}")

    lstm_row = _lstm_row(config, raga_to_label, val_tracks, test_tracks, cache_dir, pipeline_kwargs)
    print(f"  LSTM: val_macro_f1={lstm_row['val_macro_f1']:.3f}  test_macro_f1={lstm_row['test_macro_f1']:.3f}")

    rows = [
        (lstm_row["name"], lstm_row["val_macro_f1"], lstm_row["test_macro_f1"], lstm_row["num_params"], lstm_row["mean_inference_latency_ms"]),
        (svm_result.name, svm_result.val_macro_f1, svm_result.test_macro_f1, svm_result.num_params, svm_result.mean_inference_latency_ms),
        (mlp_result.name, mlp_result.val_macro_f1, mlp_result.test_macro_f1, mlp_result.num_params, mlp_result.mean_inference_latency_ms),
    ]

    val_header = f"Val macro-F1 ({len(val_tracks)} tracks)"
    test_header = f"Test macro-F1 ({len(test_tracks)} tracks)"
    lines = [
        f"| Model              | {val_header} | {test_header} | Params | Inference latency |",
        f"|--------------------|-{'-' * len(val_header)}-|-{'-' * len(test_header)}-|--------|-------------------|",
    ]
    for name, val_f1, test_f1, params, latency in rows:
        lines.append(
            f"| {name:<18s} | {val_f1:>{len(val_header)}.3f} | {test_f1:>{len(test_header)}.3f} "
            f"| {params:>6d} | {latency:>15.2f}ms |"
        )

    table = "\n".join(lines)
    print("\n" + table)

    models_dir = Path(config["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    out_path = models_dir / "baseline_comparison.md"
    out_path.write_text(table + "\n", encoding="utf-8")
    print(f"\nWrote {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
