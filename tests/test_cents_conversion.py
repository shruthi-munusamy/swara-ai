"""Tests for tonic-relative cents conversion."""

import numpy as np
import pytest

from src.pitch.cents import (
    MAX_SHRUTHI_HZ,
    MIN_SHRUTHI_HZ,
    CentsContour,
    CentsConversionError,
    fold_to_octave,
    hz_to_cents,
    to_cents_contour,
)
from src.pitch.extractor import PitchContour


# ----- Fundamental math -----

class TestHzToCents:

    def test_tonic_is_zero_cents(self):
        result = hz_to_cents(np.array([220.0]), shruthi_hz=220.0)
        assert result[0] == pytest.approx(0.0)

    def test_octave_above_is_1200_cents(self):
        result = hz_to_cents(np.array([440.0]), shruthi_hz=220.0)
        assert result[0] == pytest.approx(1200.0)

    def test_octave_below_is_negative_1200(self):
        result = hz_to_cents(np.array([110.0]), shruthi_hz=220.0)
        assert result[0] == pytest.approx(-1200.0)

    def test_semitone_is_100_cents(self):
        # 12-TET semitone ratio is 2^(1/12).
        ratio = 2.0 ** (1.0 / 12.0)
        result = hz_to_cents(np.array([220.0 * ratio]), shruthi_hz=220.0)
        assert result[0] == pytest.approx(100.0)

    def test_perfect_fifth_is_about_702_cents(self):
        """Just-intonation fifth (3/2 ratio) is ~702 cents, not 700."""
        result = hz_to_cents(np.array([220.0 * 1.5]), shruthi_hz=220.0)
        assert result[0] == pytest.approx(701.955, abs=0.01)

    def test_array_vectorized(self):
        freqs = np.array([220.0, 440.0, 880.0])
        result = hz_to_cents(freqs, shruthi_hz=220.0)
        expected = np.array([0.0, 1200.0, 2400.0])
        np.testing.assert_allclose(result, expected)

    def test_nonpositive_becomes_nan(self):
        freqs = np.array([220.0, 0.0, -1.0, 440.0])
        result = hz_to_cents(freqs, shruthi_hz=220.0)
        assert not np.isnan(result[0])
        assert np.isnan(result[1])
        assert np.isnan(result[2])
        assert not np.isnan(result[3])

    def test_shruthi_below_range_raises(self):
        with pytest.raises(CentsConversionError, match="vocal range"):
            hz_to_cents(np.array([220.0]), shruthi_hz=MIN_SHRUTHI_HZ - 1.0)

    def test_shruthi_above_range_raises(self):
        with pytest.raises(CentsConversionError, match="vocal range"):
            hz_to_cents(np.array([220.0]), shruthi_hz=MAX_SHRUTHI_HZ + 1.0)


# ----- Shruthi invariance: the key property -----

class TestShrutiInvariance:
    """
    The whole point of this module: two singers at different tonics singing
    the same raga should produce the same folded cents values.
    """

    def test_same_swara_at_different_shruthis_yields_same_cents(self):
        """
        Singer A sings G3 (5/4 ratio) at shruthi 220 Hz: 220 * 5/4 = 275 Hz.
        Singer B sings G3 at shruthi 145 Hz: 145 * 5/4 = 181.25 Hz.
        Both should map to ~386 cents (just-intonation G3).
        """
        cents_a = hz_to_cents(np.array([275.0]), shruthi_hz=220.0)
        cents_b = hz_to_cents(np.array([181.25]), shruthi_hz=145.0)
        assert cents_a[0] == pytest.approx(cents_b[0], abs=1e-6)
        assert cents_a[0] == pytest.approx(386.31, abs=0.1)

    def test_octave_folding_normalizes_register(self):
        """
        Singer at the same Sa but different octaves should land on the same
        cents after folding.
        """
        sa_lower = hz_to_cents(np.array([220.0]), shruthi_hz=220.0)
        sa_middle = hz_to_cents(np.array([440.0]), shruthi_hz=220.0)
        sa_upper = hz_to_cents(np.array([880.0]), shruthi_hz=220.0)
        np.testing.assert_allclose(
            fold_to_octave(sa_lower), fold_to_octave(sa_middle)
        )
        np.testing.assert_allclose(
            fold_to_octave(sa_middle), fold_to_octave(sa_upper)
        )


# ----- Octave folding -----

class TestFoldToOctave:

    def test_in_range_unchanged(self):
        result = fold_to_octave(np.array([0.0, 500.0, 1199.9]))
        np.testing.assert_allclose(result, np.array([0.0, 500.0, 1199.9]))

    def test_above_octave_folds(self):
        result = fold_to_octave(np.array([1200.0, 1500.0, 2400.0]))
        np.testing.assert_allclose(result, np.array([0.0, 300.0, 0.0]))

    def test_below_zero_folds(self):
        result = fold_to_octave(np.array([-100.0, -1200.0, -1500.0]))
        np.testing.assert_allclose(result, np.array([1100.0, 0.0, 900.0]))

    def test_nan_passes_through(self):
        result = fold_to_octave(np.array([np.nan, 400.0, np.nan]))
        assert np.isnan(result[0])
        assert result[1] == pytest.approx(400.0)
        assert np.isnan(result[2])

    def test_result_is_half_open_interval(self):
        """Folded values must be in [0, 1200), never exactly 1200."""
        result = fold_to_octave(np.array([1200.0, 2400.0, 3600.0]))
        assert np.all(result < 1200.0)
        assert np.all(result >= 0.0)


# ----- End-to-end contour conversion -----

def _make_pitch_contour(
    frequencies_hz,
    confidence=None,
    step_size_ms: int = 10,
) -> PitchContour:
    """Construct a PitchContour fixture without needing real audio."""
    n = len(frequencies_hz)
    times = np.arange(n) * (step_size_ms / 1000.0)
    if confidence is None:
        confidence = np.ones(n)
    return PitchContour(
        times=times,
        frequencies_hz=np.asarray(frequencies_hz, dtype=np.float64),
        confidence=np.asarray(confidence, dtype=np.float64),
        step_size_ms=step_size_ms,
        model="tiny",
    )


class TestToCentsContour:

    def test_returns_cents_contour(self):
        pc = _make_pitch_contour([220.0, 440.0])
        cc = to_cents_contour(pc, shruthi_hz=220.0, confidence_threshold=0.5)
        assert isinstance(cc, CentsContour)

    def test_preserves_times(self):
        pc = _make_pitch_contour([220.0, 440.0, 880.0])
        cc = to_cents_contour(pc, shruthi_hz=220.0, confidence_threshold=0.5)
        np.testing.assert_allclose(cc.times, pc.times)

    def test_preserves_confidence(self):
        conf = np.array([0.9, 0.3, 0.8])
        pc = _make_pitch_contour([220.0, 440.0, 880.0], confidence=conf)
        cc = to_cents_contour(pc, shruthi_hz=220.0, confidence_threshold=0.5)
        np.testing.assert_allclose(cc.confidence, conf)

    def test_preserves_step_size(self):
        pc = _make_pitch_contour([220.0, 440.0], step_size_ms=20)
        cc = to_cents_contour(pc, shruthi_hz=220.0, confidence_threshold=0.5)
        assert cc.step_size_ms == 20

    def test_records_shruthi(self):
        pc = _make_pitch_contour([220.0, 440.0])
        cc = to_cents_contour(pc, shruthi_hz=220.0, confidence_threshold=0.5)
        assert cc.shruthi_hz == 220.0

    def test_octave_fold_default_true(self):
        pc = _make_pitch_contour([220.0, 440.0])
        cc = to_cents_contour(pc, shruthi_hz=220.0, confidence_threshold=0.5)
        assert cc.octave_folded is True
        # 440 Hz at shruthi 220 = 1200 cents, folds to 0.
        assert cc.cents[1] == pytest.approx(0.0)

    def test_octave_fold_false_preserves_unfolded(self):
        pc = _make_pitch_contour([220.0, 440.0])
        cc = to_cents_contour(
            pc, shruthi_hz=220.0, confidence_threshold=0.5, octave_fold=False
        )
        assert cc.octave_folded is False
        assert cc.cents[1] == pytest.approx(1200.0)

    def test_voicing_mask_applied(self):
        conf = np.array([0.9, 0.3, 0.8])
        pc = _make_pitch_contour([220.0, 440.0, 880.0], confidence=conf)
        cc = to_cents_contour(pc, shruthi_hz=220.0, confidence_threshold=0.5)
        expected = np.array([True, False, True])
        np.testing.assert_array_equal(cc.voiced, expected)

    def test_voiced_cents_returns_only_voiced(self):
        conf = np.array([0.9, 0.3, 0.8])
        pc = _make_pitch_contour([220.0, 440.0, 880.0], confidence=conf)
        cc = to_cents_contour(pc, shruthi_hz=220.0, confidence_threshold=0.5)
        assert len(cc.voiced_cents()) == 2

    def test_voiced_times_aligns_with_voiced_cents(self):
        conf = np.array([0.9, 0.3, 0.8])
        pc = _make_pitch_contour([220.0, 440.0, 880.0], confidence=conf)
        cc = to_cents_contour(pc, shruthi_hz=220.0, confidence_threshold=0.5)
        assert len(cc.voiced_times()) == len(cc.voiced_cents())

    def test_empty_contour_raises(self):
        pc = _make_pitch_contour([])
        with pytest.raises(CentsConversionError, match="empty"):
            to_cents_contour(pc, shruthi_hz=220.0, confidence_threshold=0.5)