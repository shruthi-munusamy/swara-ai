"""Tests for the identifier evaluation harness."""

from pathlib import Path

import numpy as np
import pytest
import torch

from src.identifier.dataset import NUM_SWARA_CLASSES, TrackInfo, _cache_path
from src.identifier.evaluate import (
    EvaluationReport,
    TrackPrediction,
    allied_raga_overlap,
    evaluate_predictions,
    predict_tracks,
    top_confusion_pairs,
)
from src.identifier.model import RagaLSTM

PIPELINE_KWARGS = dict(
    sample_rate=16000, crepe_model="tiny", step_size_ms=10,
    confidence_threshold=0.5, tolerance_cents=35.0, min_stable_ms=80.0, smoothing_window=5,
)


def _seed_track(cache_dir: Path, track_id: str, raga: str, sequence) -> TrackInfo:
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(_cache_path(cache_dir, track_id), swara_indices=np.asarray(sequence, dtype=np.int64))
    return TrackInfo(track_id, Path("unused.wav"), 220.0, raga, 60.0)


class TestPredictTracks:

    def test_returns_one_prediction_per_track(self, tmp_path):
        cache_dir = tmp_path / "cache"
        t1 = _seed_track(cache_dir, "t1", "Todi", [0, 1, 2] * 5)
        t2 = _seed_track(cache_dir, "t2", "Kalyani", [5, 6, 7] * 5)
        model = RagaLSTM(NUM_SWARA_CLASSES, num_ragas=2, embedding_dim=8, hidden_size=16, num_layers=2, dropout=0.0)

        preds = predict_tracks(
            model, [t1, t2], {"Todi": 0, "Kalyani": 1}, cache_dir, PIPELINE_KWARGS,
            window_size=10, window_stride=5, device=torch.device("cpu"),
        )
        assert len(preds) == 2
        assert {p.track_id for p in preds} == {"t1", "t2"}

    def test_true_label_matches_raga_mapping(self, tmp_path):
        cache_dir = tmp_path / "cache"
        t1 = _seed_track(cache_dir, "t1", "Kalyani", [0, 1, 2] * 5)
        model = RagaLSTM(NUM_SWARA_CLASSES, num_ragas=2, embedding_dim=8, hidden_size=16, num_layers=2, dropout=0.0)

        preds = predict_tracks(
            model, [t1], {"Todi": 0, "Kalyani": 1}, cache_dir, PIPELINE_KWARGS,
            window_size=10, window_stride=5, device=torch.device("cpu"),
        )
        assert preds[0].true_label == 1

    def test_top3_confidences_sum_to_at_most_one(self, tmp_path):
        cache_dir = tmp_path / "cache"
        t1 = _seed_track(cache_dir, "t1", "Todi", [0, 1, 2] * 5)
        model = RagaLSTM(NUM_SWARA_CLASSES, num_ragas=4, embedding_dim=8, hidden_size=16, num_layers=2, dropout=0.0)

        preds = predict_tracks(
            model, [t1], {"Todi": 0, "A": 1, "B": 2, "C": 3}, cache_dir, PIPELINE_KWARGS,
            window_size=10, window_stride=5, device=torch.device("cpu"),
        )
        assert sum(preds[0].top3_confidences) <= 1.0 + 1e-5
        assert len(preds[0].top3_labels) == 3

    def test_track_with_zero_segments_excluded(self, tmp_path):
        cache_dir = tmp_path / "cache"
        t1 = _seed_track(cache_dir, "t1", "Todi", [])
        model = RagaLSTM(NUM_SWARA_CLASSES, num_ragas=1, embedding_dim=8, hidden_size=16, num_layers=2, dropout=0.0)

        preds = predict_tracks(
            model, [t1], {"Todi": 0}, cache_dir, PIPELINE_KWARGS,
            window_size=10, window_stride=5, device=torch.device("cpu"),
        )
        assert preds == []


class TestEvaluatePredictions:

    def _pred(self, true_label, predicted_label):
        return TrackPrediction(
            track_id=f"t{true_label}{predicted_label}", true_label=true_label,
            predicted_label=predicted_label, confidence=0.9,
            top3_labels=(predicted_label,), top3_confidences=(0.9,),
        )

    def test_perfect_predictions_give_f1_one(self):
        preds = [self._pred(0, 0), self._pred(1, 1), self._pred(0, 0)]
        report = evaluate_predictions(preds, ["Todi", "Kalyani"])
        assert report.macro_f1 == pytest.approx(1.0)
        assert report.accuracy == pytest.approx(1.0)

    def test_confusion_matrix_shape_and_content(self):
        preds = [self._pred(0, 1), self._pred(0, 1), self._pred(1, 1)]
        report = evaluate_predictions(preds, ["Todi", "Kalyani"])
        assert report.confusion.shape == (2, 2)
        assert report.confusion[0, 1] == 2  # Todi mistaken for Kalyani twice
        assert report.confusion[1, 1] == 1

    def test_per_raga_metrics_keyed_by_name(self):
        preds = [self._pred(0, 0), self._pred(1, 0)]
        report = evaluate_predictions(preds, ["Todi", "Kalyani"])
        assert set(report.per_raga_f1.keys()) == {"Todi", "Kalyani"}


class TestTopConfusionPairs:

    def test_sorted_by_count_descending(self):
        cm = np.array([[5, 3, 0], [1, 10, 2], [0, 0, 4]])
        report = EvaluationReport(
            predictions=[], label_names=["A", "B", "C"],
            per_raga_precision={}, per_raga_recall={}, per_raga_f1={},
            macro_f1=0.0, weighted_f1=0.0, accuracy=0.0, confusion=cm,
        )
        pairs = top_confusion_pairs(report, top_k=3)
        assert pairs[0] == ("A", "B", 3)
        assert ("B", "A", 1) in pairs
        assert ("B", "C", 2) in pairs

    def test_no_off_diagonal_confusions_returns_empty(self):
        cm = np.array([[5, 0], [0, 5]])
        report = EvaluationReport(
            predictions=[], label_names=["A", "B"],
            per_raga_precision={}, per_raga_recall={}, per_raga_f1={},
            macro_f1=1.0, weighted_f1=1.0, accuracy=1.0, confusion=cm,
        )
        assert top_confusion_pairs(report) == []


class TestAlliedRagaOverlap:

    def test_resolves_known_melakartas(self):
        # Sankarabharanam (mela 29) and Kalyani (mela 65) differ by one
        # swara (M1 vs M2) -- 6 of 7 positions shared.
        result = allied_raga_overlap("Sankarabharanam", "Kalyani")
        assert "shared" in result
        assert "not verified" not in result

    def test_unresolvable_janya_raga_reports_not_verified(self):
        result = allied_raga_overlap("Bhairavi", "Kalyani")
        assert "not verified" in result
