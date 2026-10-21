"""
Train and evaluate the SVM/MLP baselines, and produce a comparison table
against the LSTM identifier.

Prerequisites:
    python scripts/prepare_identifier_data.py
    python scripts/train_identifier.py
    python scripts/evaluate_identifier.py   (for the LSTM's test macro-F1)

Usage:
    python scripts/evaluate_baselines.py

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
from src.identifier.model import RagaLSTM

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
DEFAULT_SARAGA_ROOT = Path("data/raw/saraga/saraga1.5_carnatic")
DEFAULT_PROCESSED_DIR = Path("data/processed")


def _load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _lstm_row(config: dict, raga_to_label: dict, test_macro_f1_from_evaluate: float) -> dict:
    """
    Re-derive the LSTM's param count and single-window inference latency
    for the comparison table. The test macro-F1 itself comes from
    scripts/evaluate_identifier.py's output rather than being recomputed
    here, so there's exactly one code path that produces that number.
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
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

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
        "val_macro_f1": None,  # not recomputed here; see models/identifier_best.pt training log
        "test_macro_f1": test_macro_f1_from_evaluate,
        "num_params": num_params,
        "mean_inference_latency_ms": latency_ms,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--saraga-root", type=Path, default=DEFAULT_SARAGA_ROOT)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument(
        "--lstm-test-macro-f1", type=float, required=True,
        help="Test macro-F1 printed by scripts/evaluate_identifier.py (not recomputed here).",
    )
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

    lstm_row = _lstm_row(config, raga_to_label, args.lstm_test_macro_f1)

    rows = [
        (svm_result.name, svm_result.val_macro_f1, svm_result.test_macro_f1, svm_result.num_params, svm_result.mean_inference_latency_ms),
        (mlp_result.name, mlp_result.val_macro_f1, mlp_result.test_macro_f1, mlp_result.num_params, mlp_result.mean_inference_latency_ms),
        (lstm_row["name"], None, lstm_row["test_macro_f1"], lstm_row["num_params"], lstm_row["mean_inference_latency_ms"]),
    ]

    lines = [
        "| Model              | Val macro-F1 | Test macro-F1 | Params | Inference latency |",
        "|--------------------|--------------|---------------|--------|--------------------|",
    ]
    for name, val_f1, test_f1, params, latency in rows:
        val_str = f"{val_f1:.3f}" if val_f1 is not None else "n/a*"
        lines.append(f"| {name:<18s} | {val_str:>12s} | {test_f1:>13.3f} | {params:>6d} | {latency:>14.2f}ms |")
    lines.append("")
    lines.append(
        "*LSTM val macro-F1 (0.595, best epoch 13) comes from the training run "
        "log, not recomputed by this script -- see README training section."
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
