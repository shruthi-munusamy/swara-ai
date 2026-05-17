"""Tests for swara segmentation."""

import numpy as np
import pytest

from src.pitch.cents import CentsContour
from src.swara.segmenter import (
    SegmentationError,
    SwaraSegment,
    segment_swaras,
)


def _make_cents_contour(
    cents_values,
    voiced=None,
    step_size_ms: int = 10,
    octave_folded: bool = True,
) -> CentsContour:
    """Construct a CentsContour fixture from raw arrays."""
    n = len(cents_values)
    times = np.arange(n) * (step_size_ms / 1000.0)
    cents = np.asarray(cents_values, dtype=np.float64)
    if voiced is None:
        voiced = np.ones(n, dtype=bool)
    confidence = np.where(voiced, 0.9, 0.1).astype(np.float64)
    return CentsContour(
        times=times,
        cents=cents,
        confidence=confidence,
        voiced=np.asarray(voiced, dtype=bool),
        shruthi_hz=220.0,
        octave_folded=octave_folded,
        step_size_ms=step_size_ms,
    )


# ----- Sanity / contract -----

class TestBasicSegmentation:

    def test_returns_list(self):
        cents = [0.0] * 50
        cc = _make_cents_contour(cents)
        result = segment_swaras(cc, tolerance_cents=35.0, min_stable_ms=80.0, smoothing_window=5)
        assert isinstance(result, list)

    def test_empty_contour_returns_empty_list(self):
        cc = _make_cents_contour([])
        result = segment_swaras(cc, tolerance_cents=35.0, min_stable_ms=80.0, smoothing_window=5)
        assert result == []

    def test_all_unvoiced_returns_empty_list(self):
        cents = [0.0] * 50
        voiced = [False] * 50
        cc = _make_cents_contour(cents, voiced=voiced)
        result = segment_swaras(cc, tolerance_cents=35.0, min_stable_ms=80.0, smoothing_window=5)
        assert result == []


# ----- Detection accuracy -----

class TestSwaraDetection:

    def test_detects_sustained_sa(self):
        """A 500ms hold at 0 cents should produce one Sa segment."""
        cents = [0.0] * 50  # 50 frames * 10ms = 500ms
        cc = _make_cents_contour(cents)
        segments = segment_swaras(cc, tolerance_cents=35.0, min_stable_ms=80.0, smoothing_window=5)
        assert len(segments) == 1
        assert "S" in segments[0].swara_candidates

    def test_detects_sequence_of_swaras(self):
        """Sa → R1 → G3 → P, each held 200ms."""
        frames_per_swara = 20
        cents = (
            [0.0] * frames_per_swara
            + [100.0] * frames_per_swara
            + [400.0] * frames_per_swara
            + [700.0] * frames_per_swara
        )
        cc = _make_cents_contour(cents)
        segments = segment_swaras(cc, tolerance_cents=35.0, min_stable_ms=80.0, smoothing_window=5)
        assert len(segments) == 4
        assert "S" in segments[0].swara_candidates
        assert "R1" in segments[1].swara_candidates
        assert "G3" in segments[2].swara_candidates
        assert "P" in segments[3].swara_candidates

    def test_shared_cents_position_returns_both_names(self):
        """200 cents = R2 = G1. Both should appear in candidates."""
        cents = [200.0] * 30
        cc = _make_cents_contour(cents)
        segments = segment_swaras(cc, tolerance_cents=35.0, min_stable_ms=80.0, smoothing_window=5)
        assert len(segments) == 1
        assert "R2" in segments[0].swara_candidates
        assert "G1" in segments[0].swara_candidates


# ----- Duration filtering -----

class TestDurationFiltering:

    def test_drops_too_short_segments(self):
        """40ms Sa is below the 80ms threshold and should be discarded."""
        cents = [0.0] * 4  # 4 frames * 10ms = 40ms
        cc = _make_cents_contour(cents)
        segments = segment_swaras(cc, tolerance_cents=35.0, min_stable_ms=80.0, smoothing_window=1)
        assert segments == []

    def test_keeps_long_enough_segments(self):
        cents = [0.0] * 10  # 100ms — just over threshold
        cc = _make_cents_contour(cents)
        segments = segment_swaras(cc, tolerance_cents=35.0, min_stable_ms=80.0, smoothing_window=1)
        assert len(segments) == 1


# ----- Transitions and noise -----

class TestTransitions:

    def test_out_of_tolerance_frames_excluded(self):
        """350 cents is between G2/R3 (300c) and G3 (400c); outside ±35c of either."""
        cents = [350.0] * 30
        cc = _make_cents_contour(cents)
        segments = segment_swaras(cc, tolerance_cents=35.0, min_stable_ms=80.0, smoothing_window=1)
        assert segments == []

    def test_split_by_transition_region(self):
        """Sa, then a transition (out of tolerance), then P."""
        cents = (
            [0.0] * 20      # Sa, 200ms
            + [350.0] * 5   # transition, 50ms — too short and out of tolerance anyway
            + [700.0] * 20  # P, 200ms
        )
        cc = _make_cents_contour(cents)
        segments = segment_swaras(cc, tolerance_cents=35.0, min_stable_ms=80.0, smoothing_window=1)
        assert len(segments) == 2
        assert "S" in segments[0].swara_candidates
        assert "P" in segments[1].swara_candidates


# ----- Smoothing -----

class TestSmoothing:

    def test_smoothing_suppresses_single_frame_blip(self):
        """A single-frame Ri blip in the middle of Sa should be smoothed out."""
        cents = [0.0] * 10 + [100.0] + [0.0] * 10
        cc = _make_cents_contour(cents)
        segments = segment_swaras(cc, tolerance_cents=35.0, min_stable_ms=80.0, smoothing_window=5)
        # Without smoothing this would be Sa / R1 / Sa; with median window 5,
        # the single R1 frame is overwhelmed and we get one Sa.
        assert len(segments) == 1
        assert "S" in segments[0].swara_candidates

    def test_even_smoothing_window_raises(self):
        cc = _make_cents_contour([0.0] * 30)
        with pytest.raises(SegmentationError, match="odd"):
            segment_swaras(cc, tolerance_cents=35.0, min_stable_ms=80.0, smoothing_window=4)

    def test_zero_smoothing_window_raises(self):
        cc = _make_cents_contour([0.0] * 30)
        with pytest.raises(SegmentationError):
            segment_swaras(cc, tolerance_cents=35.0, min_stable_ms=80.0, smoothing_window=0)


# ----- Input validation -----

class TestInputValidation:

    def test_unfolded_contour_raises(self):
        cc = _make_cents_contour([0.0] * 30, octave_folded=False)
        with pytest.raises(SegmentationError, match="octave-folded"):
            segment_swaras(cc, tolerance_cents=35.0, min_stable_ms=80.0, smoothing_window=5)

    def test_negative_tolerance_raises(self):
        cc = _make_cents_contour([0.0] * 30)
        with pytest.raises(SegmentationError, match="tolerance"):
            segment_swaras(cc, tolerance_cents=-1.0, min_stable_ms=80.0, smoothing_window=5)

    def test_negative_min_stable_raises(self):
        cc = _make_cents_contour([0.0] * 30)
        with pytest.raises(SegmentationError, match="min_stable"):
            segment_swaras(cc, tolerance_cents=35.0, min_stable_ms=-1.0, smoothing_window=5)


# ----- SwaraSegment properties -----

class TestSwaraSegment:

    def test_duration_seconds_property(self):
        cents = [0.0] * 30
        cc = _make_cents_contour(cents)
        segments = segment_swaras(cc, tolerance_cents=35.0, min_stable_ms=80.0, smoothing_window=1)
        seg = segments[0]
        assert seg.duration_seconds == pytest.approx(seg.end_time - seg.start_time)

    def test_primary_candidate_is_alphabetically_first(self):
        """200 cents has R2 and G1; sorted alphabetically, G1 comes first."""
        cents = [200.0] * 30
        cc = _make_cents_contour(cents)
        segments = segment_swaras(cc, tolerance_cents=35.0, min_stable_ms=80.0, smoothing_window=1)
        assert segments[0].primary_candidate == "G1"

    def test_segments_are_time_ordered(self):
        cents = [0.0] * 20 + [400.0] * 20 + [700.0] * 20
        cc = _make_cents_contour(cents)
        segments = segment_swaras(cc, tolerance_cents=35.0, min_stable_ms=80.0, smoothing_window=1)
        times = [s.start_time for s in segments]
        assert times == sorted(times)

    def test_confidence_averaged_across_frames(self):
        cents = [0.0] * 30
        cc = _make_cents_contour(cents)
        segments = segment_swaras(cc, tolerance_cents=35.0, min_stable_ms=80.0, smoothing_window=1)
        # All voiced frames have confidence 0.9; the segment confidence should match.
        assert segments[0].confidence == pytest.approx(0.9)