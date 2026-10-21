"""
Baseline models for raga identification: SVM on swara histograms, MLP on
pooled swara-position statistics. Same windowed train/val/test examples as
the LSTM (src/identifier/dataset.py), so the comparison is apples-to-apples.

Why these two baselines:
    - SVM on histograms: the simplest model that could possibly work --
      "what fraction of the clip was spent on each swara," with zero
      sequence information. If the LSTM barely beats this, the sequence
      modeling isn't earning its complexity.
    - MLP on mean/std/max-pooled one-hot swara vectors: adds basic
      distributional shape (dispersion, peak usage) beyond a flat
      histogram, still with no explicit sequence order.

Both are trained and evaluated on the same windows as the LSTM, then
aggregated to one prediction per track via majority vote across a track's
windows -- not probability averaging, since sklearn's SVC(probability=True)
calibration is slow and majority vote is simpler and still a fair,
deterministic aggregation rule.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from src.identifier.dataset import NUM_SWARA_CLASSES, SaragaSwaraDataset, TrackInfo


def histogram_features(sequence: np.ndarray) -> np.ndarray:
    """Normalized 12-bin histogram of swara class occurrences in a window."""
    counts = np.bincount(sequence, minlength=NUM_SWARA_CLASSES).astype(np.float64)
    total = counts.sum()
    return counts / total if total > 0 else counts


def aggregate_features(sequence: np.ndarray) -> np.ndarray:
    """36-dim: mean, std, and max of one-hot swara vectors pooled over a window."""
    one_hot = np.eye(NUM_SWARA_CLASSES)[sequence]  # (window_len, 12)
    return np.concatenate([one_hot.mean(axis=0), one_hot.std(axis=0), one_hot.max(axis=0)])


def _build_windowed_matrix(
    tracks: Sequence[TrackInfo],
    raga_to_label: Dict[str, int],
    cache_dir: Path,
    pipeline_kwargs: dict,
    window_size: int,
    window_stride: int,
    feature_fn,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Returns (X, y, track_indices): one row per window, feature_fn applied
    to each window's swara sequence, plus which track (index into `tracks`)
    each row came from -- needed to aggregate back to track level.
    """
    ds = SaragaSwaraDataset(tracks, raga_to_label, cache_dir, pipeline_kwargs, window_size, window_stride)
    labels = ds.labels()
    track_indices = ds.track_indices()
    X = np.stack([feature_fn(ds[i][0].numpy()) for i in range(len(ds))]) if len(ds) > 0 else np.empty((0, 0))
    y = np.asarray(labels, dtype=np.int64)
    return X, y, track_indices


def _majority_vote_by_track(
    window_preds: np.ndarray, track_indices: Sequence[int], num_tracks: int
) -> np.ndarray:
    """Aggregate window-level predicted labels to one label per track via majority vote."""
    track_indices_arr = np.asarray(track_indices)
    track_preds = np.full(num_tracks, -1, dtype=np.int64)
    for t in range(num_tracks):
        mask = track_indices_arr == t
        if not mask.any():
            continue
        votes = window_preds[mask]
        values, counts = np.unique(votes, return_counts=True)
        track_preds[t] = values[np.argmax(counts)]
    return track_preds


@dataclass(frozen=True)
class BaselineResult:
    name: str
    val_macro_f1: float
    test_macro_f1: float
    num_params: int
    mean_inference_latency_ms: float


def _track_labels(tracks: Sequence[TrackInfo], raga_to_label: Dict[str, int]) -> np.ndarray:
    return np.asarray([raga_to_label[t.raga] for t in tracks], dtype=np.int64)


def _evaluate_track_level(
    model, X: np.ndarray, track_indices: List[int], tracks: Sequence[TrackInfo], raga_to_label: Dict[str, int]
) -> float:
    if X.shape[0] == 0:
        return 0.0
    window_preds = model.predict(X)
    track_preds = _majority_vote_by_track(window_preds, track_indices, len(tracks))
    true_labels = _track_labels(tracks, raga_to_label)
    resolved = track_preds != -1
    if not resolved.any():
        return 0.0
    return f1_score(true_labels[resolved], track_preds[resolved], average="macro", zero_division=0)


def _measure_latency_ms(model, x_single: np.ndarray, repeats: int = 20) -> float:
    x_single = x_single.reshape(1, -1)
    start = time.perf_counter()
    for _ in range(repeats):
        model.predict(x_single)
    elapsed = time.perf_counter() - start
    return (elapsed / repeats) * 1000.0


def train_svm_baseline(
    train_tracks: Sequence[TrackInfo],
    val_tracks: Sequence[TrackInfo],
    test_tracks: Sequence[TrackInfo],
    raga_to_label: Dict[str, int],
    cache_dir: Path,
    pipeline_kwargs: dict,
    window_size: int,
    window_stride: int,
) -> BaselineResult:
    """RBF-kernel SVM on normalized swara histograms, C/gamma grid-searched on val."""
    X_train, y_train, _ = _build_windowed_matrix(
        train_tracks, raga_to_label, cache_dir, pipeline_kwargs, window_size, window_stride, histogram_features
    )
    X_val, y_val, val_idx = _build_windowed_matrix(
        val_tracks, raga_to_label, cache_dir, pipeline_kwargs, window_size, window_stride, histogram_features
    )
    X_test, y_test, test_idx = _build_windowed_matrix(
        test_tracks, raga_to_label, cache_dir, pipeline_kwargs, window_size, window_stride, histogram_features
    )

    grid = GridSearchCV(
        SVC(kernel="rbf"),
        param_grid={"C": [0.1, 1.0, 10.0], "gamma": ["scale", 0.1, 1.0]},
        cv=3,
    )
    grid.fit(X_train, y_train)
    model = grid.best_estimator_

    val_macro_f1 = _evaluate_track_level(model, X_val, val_idx, val_tracks, raga_to_label)
    test_macro_f1 = _evaluate_track_level(model, X_test, test_idx, test_tracks, raga_to_label)
    latency = _measure_latency_ms(model, X_test[0]) if X_test.shape[0] > 0 else float("nan")

    return BaselineResult(
        name="SVM (histograms)",
        val_macro_f1=val_macro_f1,
        test_macro_f1=test_macro_f1,
        num_params=int(model.support_vectors_.size),
        mean_inference_latency_ms=latency,
    )


def train_mlp_baseline(
    train_tracks: Sequence[TrackInfo],
    val_tracks: Sequence[TrackInfo],
    test_tracks: Sequence[TrackInfo],
    raga_to_label: Dict[str, int],
    cache_dir: Path,
    pipeline_kwargs: dict,
    window_size: int,
    window_stride: int,
    seed: int = 42,
) -> BaselineResult:
    """2-hidden-layer MLP on mean/std/max-pooled one-hot swara vectors."""
    X_train, y_train, _ = _build_windowed_matrix(
        train_tracks, raga_to_label, cache_dir, pipeline_kwargs, window_size, window_stride, aggregate_features
    )
    X_val, y_val, val_idx = _build_windowed_matrix(
        val_tracks, raga_to_label, cache_dir, pipeline_kwargs, window_size, window_stride, aggregate_features
    )
    X_test, y_test, test_idx = _build_windowed_matrix(
        test_tracks, raga_to_label, cache_dir, pipeline_kwargs, window_size, window_stride, aggregate_features
    )

    model = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        max_iter=500,
        random_state=seed,
        early_stopping=True,
    )
    model.fit(X_train, y_train)

    val_macro_f1 = _evaluate_track_level(model, X_val, val_idx, val_tracks, raga_to_label)
    test_macro_f1 = _evaluate_track_level(model, X_test, test_idx, test_tracks, raga_to_label)
    latency = _measure_latency_ms(model, X_test[0]) if X_test.shape[0] > 0 else float("nan")

    num_params = sum(w.size for w in model.coefs_) + sum(b.size for b in model.intercepts_)

    return BaselineResult(
        name="MLP (aggregates)",
        val_macro_f1=val_macro_f1,
        test_macro_f1=test_macro_f1,
        num_params=int(num_params),
        mean_inference_latency_ms=latency,
    )
