"""Tests for the SVM/MLP raga identification baselines."""

from pathlib import Path

import numpy as np
import pytest

from src.identifier.baselines import (
    _majority_vote_by_track,
    aggregate_features,
    histogram_features,
    train_mlp_baseline,
    train_svm_baseline,
)
from src.identifier.dataset import NUM_SWARA_CLASSES, TrackInfo, _cache_path

PIPELINE_KWARGS = dict(
    sample_rate=16000, crepe_model="tiny", step_size_ms=10,
    confidence_threshold=0.5, tolerance_cents=35.0, min_stable_ms=80.0, smoothing_window=5,
)


class TestHistogramFeatures:

    def test_sums_to_one(self):
        seq = np.array([0, 0, 1, 2, 2, 2])
        feats = histogram_features(seq)
        assert feats.sum() == pytest.approx(1.0)
        assert feats.shape == (NUM_SWARA_CLASSES,)

    def test_empty_sequence_is_all_zero(self):
        feats = histogram_features(np.array([], dtype=np.int64))
        assert np.all(feats == 0.0)


class TestAggregateFeatures:

    def test_shape_is_36(self):
        seq = np.array([0, 1, 2, 3])
        feats = aggregate_features(seq)
        assert feats.shape == (NUM_SWARA_CLASSES * 3,)

    def test_mean_component_matches_histogram(self):
        seq = np.array([0, 0, 1, 2])
        feats = aggregate_features(seq)
        hist = histogram_features(seq)
        assert np.allclose(feats[:NUM_SWARA_CLASSES], hist)


class TestMajorityVoteByTrack:

    def test_majority_wins(self):
        preds = np.array([0, 0, 1, 2, 2, 2])
        track_idx = [0, 0, 0, 1, 1, 1]
        result = _majority_vote_by_track(preds, track_idx, num_tracks=2)
        assert result[0] == 0
        assert result[1] == 2

    def test_track_with_no_windows_gets_sentinel(self):
        preds = np.array([0, 0])
        track_idx = [0, 0]
        result = _majority_vote_by_track(preds, track_idx, num_tracks=2)
        assert result[1] == -1


def _seed_track(cache_dir: Path, track_id: str, raga: str, sequence) -> TrackInfo:
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(_cache_path(cache_dir, track_id), swara_indices=np.asarray(sequence, dtype=np.int64))
    return TrackInfo(track_id, Path("unused.wav"), 220.0, raga, 60.0)


class TestBaselineTrainingSmoke:
    """End-to-end smoke tests on tiny seeded data -- not accuracy claims."""

    def _make_tracks(self, cache_dir):
        # Two trivially separable classes: all-low vs all-high swara indices.
        tracks = []
        for i in range(4):
            tracks.append(_seed_track(cache_dir, f"todi_{i}", "Todi", [0, 1, 2] * 20))
        for i in range(4):
            tracks.append(_seed_track(cache_dir, f"kalyani_{i}", "Kalyani", [9, 10, 11] * 20))
        return tracks

    def test_svm_baseline_runs_and_returns_result(self, tmp_path):
        # tracks[0:4] are Todi, tracks[4:8] are Kalyani -- every split must
        # include both classes, or SVC.fit has nothing to discriminate.
        cache_dir = tmp_path / "cache"
        tracks = self._make_tracks(cache_dir)
        train = [tracks[0], tracks[1], tracks[4], tracks[5]]
        val = [tracks[2], tracks[6]]
        test = [tracks[3], tracks[7]]
        result = train_svm_baseline(
            train, val, test, {"Todi": 0, "Kalyani": 1},
            cache_dir, PIPELINE_KWARGS, window_size=10, window_stride=5,
        )
        assert result.name == "SVM (histograms)"
        assert 0.0 <= result.val_macro_f1 <= 1.0
        assert 0.0 <= result.test_macro_f1 <= 1.0
        assert result.num_params > 0
        assert result.mean_inference_latency_ms >= 0.0

    def test_mlp_baseline_runs_and_returns_result(self, tmp_path):
        cache_dir = tmp_path / "cache"
        tracks = self._make_tracks(cache_dir)
        train = [tracks[0], tracks[1], tracks[4], tracks[5]]
        val = [tracks[2], tracks[6]]
        test = [tracks[3], tracks[7]]
        result = train_mlp_baseline(
            train, val, test, {"Todi": 0, "Kalyani": 1},
            cache_dir, PIPELINE_KWARGS, window_size=10, window_stride=5,
        )
        assert result.name == "MLP (aggregates)"
        assert 0.0 <= result.val_macro_f1 <= 1.0
        assert 0.0 <= result.test_macro_f1 <= 1.0
        assert result.num_params > 0

    def test_svm_separates_trivially_distinct_classes(self, tmp_path):
        # Sanity check: on two classes with disjoint swara vocabularies,
        # the histogram SVM should achieve perfect test macro-F1.
        cache_dir = tmp_path / "cache"
        tracks = self._make_tracks(cache_dir)
        train = [tracks[0], tracks[1], tracks[2], tracks[4], tracks[5], tracks[6]]
        val = [tracks[3], tracks[7]]
        test = [tracks[3], tracks[7]]
        result = train_svm_baseline(
            train, val, test, {"Todi": 0, "Kalyani": 1},
            cache_dir, PIPELINE_KWARGS, window_size=10, window_stride=5,
        )
        assert result.test_macro_f1 == pytest.approx(1.0)
