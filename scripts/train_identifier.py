"""
Train the LSTM raga identifier on the Saraga MVP raga subset.

Prerequisites:
    python scripts/download_saraga.py
    python scripts/prepare_identifier_data.py

Usage:
    python scripts/train_identifier.py
    python scripts/train_identifier.py --wandb
    python scripts/train_identifier.py --config config.yaml --seed 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.identifier.dataset import (
    NUM_SWARA_CLASSES,
    SaragaSwaraDataset,
    TrackInfo,
    collate_fn,
    discover_tracks,
)
from src.identifier.model import RagaLSTM
from src.identifier.train import WandbLogger, set_seed, train

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
DEFAULT_SARAGA_ROOT = Path("data/raw/saraga")
DEFAULT_PROCESSED_DIR = Path("data/processed")


def _load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_tracks_and_splits(
    saraga_root: Path, processed_dir: Path
) -> Tuple[List[TrackInfo], List[TrackInfo], List[TrackInfo], Dict[str, int]]:
    """
    Reload the reproducible split produced by prepare_identifier_data.py,
    resolving each stored track_id back to a full TrackInfo via a fresh
    directory scan (TrackInfo itself isn't serialized -- only ids are --
    so the split file stays valid even if paths on disk move).
    """
    mvp_path = processed_dir / "mvp_ragas.yaml"
    splits_path = processed_dir / "splits.yaml"
    if not mvp_path.exists() or not splits_path.exists():
        raise FileNotFoundError(
            f"Missing {mvp_path} or {splits_path}. Run "
            f"scripts/prepare_identifier_data.py first."
        )

    with open(mvp_path, "r", encoding="utf-8") as f:
        mvp = yaml.safe_load(f)
    with open(splits_path, "r", encoding="utf-8") as f:
        splits = yaml.safe_load(f)

    raga_to_label = {name: i for i, name in enumerate(mvp["ragas"])}

    tracks_by_id = {t.track_id: t for t in discover_tracks(saraga_root)}

    def _resolve(track_ids: List[str]) -> List[TrackInfo]:
        missing = [tid for tid in track_ids if tid not in tracks_by_id]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} track_id(s) from splits.yaml no longer resolve "
                f"against {saraga_root} (e.g. {missing[0]!r}). Re-run "
                f"prepare_identifier_data.py if the raw data changed."
            )
        return [tracks_by_id[tid] for tid in track_ids]

    return (
        _resolve(splits["train"]),
        _resolve(splits["val"]),
        _resolve(splits["test"]),
        raga_to_label,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--saraga-root", type=Path, default=DEFAULT_SARAGA_ROOT)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--wandb", action="store_true", help="Force-enable W&B logging (overrides config.yaml).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = _load_config(args.config)
    id_cfg = config["identifier"]
    wandb_cfg = config["wandb"]
    audio_cfg = config["audio"]
    pitch_cfg = config["pitch"]
    swara_cfg = config["swara"]

    train_tracks, val_tracks, _test_tracks, raga_to_label = _load_tracks_and_splits(
        args.saraga_root, args.processed_dir
    )
    print(f"train={len(train_tracks)}  val={len(val_tracks)}  ragas={len(raga_to_label)}")

    pipeline_kwargs = dict(
        sample_rate=audio_cfg["sample_rate"],
        crepe_model=pitch_cfg["crepe_model"],
        step_size_ms=pitch_cfg["step_size_ms"],
        confidence_threshold=pitch_cfg["confidence_threshold"],
        tolerance_cents=swara_cfg["tolerance_cents"],
        min_stable_ms=swara_cfg["min_stable_ms"],
        smoothing_window=swara_cfg["smoothing_window"],
    )
    cache_dir = Path(config["paths"]["data_dir"]) / "processed" / "features"
    window_size = id_cfg["window_size_segments"]
    window_stride = id_cfg["window_stride_segments"]

    train_ds = SaragaSwaraDataset(
        train_tracks, raga_to_label, cache_dir, pipeline_kwargs, window_size, window_stride
    )
    val_ds = SaragaSwaraDataset(
        val_tracks, raga_to_label, cache_dir, pipeline_kwargs, window_size, window_stride
    )
    print(f"windowed examples: train={len(train_ds)}  val={len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=id_cfg["batch_size"], shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=id_cfg["batch_size"], shuffle=False, collate_fn=collate_fn
    )

    set_seed(args.seed)
    model = RagaLSTM(
        num_swara_classes=NUM_SWARA_CLASSES,
        num_ragas=len(raga_to_label),
        embedding_dim=id_cfg["embedding_dim"],
        hidden_size=id_cfg["hidden_size"],
        num_layers=id_cfg["num_layers"],
        dropout=id_cfg["dropout"],
    )

    wandb_logger = WandbLogger(
        enabled=args.wandb or wandb_cfg["enabled"],
        project=wandb_cfg["project"],
        entity=wandb_cfg["entity"],
        config=id_cfg,
    )

    checkpoint_path = Path(config["paths"]["models_dir"]) / "identifier_best.pt"
    train_labels = train_ds.labels()

    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        train_labels=train_labels,
        num_classes=len(raga_to_label),
        num_epochs=id_cfg["epochs"],
        learning_rate=id_cfg["learning_rate"],
        checkpoint_path=checkpoint_path,
        early_stopping_patience=id_cfg["early_stopping_patience"],
        wandb_logger=wandb_logger,
        seed=args.seed,
    )

    print(
        f"\nBest epoch: {history.best_epoch + 1 if history.best_epoch is not None else 'n/a'}  "
        f"best val_macro_f1={max(history.val_macro_f1):.4f}  "
        f"checkpoint={checkpoint_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
