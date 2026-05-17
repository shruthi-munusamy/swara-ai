"""Tests for CREPE-based pitch extraction."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.audio.loader import load_audio
from src.pitch.extractor import (
    CREPE_SAMPLE_RATE,
    PitchContour,
    PitchExtractionError,
    extract_pitch,
)


# ----- Synthetic audio fixtures -----

def _write_sine(path: Path, freq_hz: float, duration_s: float = 1.5) -> Path:
    """Write a steady sine at `freq_hz` for `duration_s` seconds at CREPE's SR."""
    n = int(duration_s * CREPE_SAMPLE_RATE)
    t = np.arange(n) / CREPE_SAMPLE_RATE
    wave = (0.5 * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float32)
    sf.write(str(path), wave, CREPE_SAMPLE_RATE)
    return path


@pytest.fixture(scope="module")
def sine_a4(tmp_path_factory) -> Path:
    """1.5 sec A4 sine (440 Hz). A4 is well within CREPE's training range."""
    path = tmp_path_factory.mktemp("audio") / "sine_a4.wav"
    return _write_sine(path, 440.0)


@pytest.fixture(scope="module")
def sine_c4(tmp_path_factory) -> Path:
    """1.5 sec C4 sine (261.63 Hz). Approximates a male vocal Sa."""
    path = tmp_path_factory.mktemp("audio") / "sine_c4.wav"
    return _write_sine(path, 261.63)


# ----- Output shape and contract -----

class TestPitchContourShape:

    def test_returns_pitch_contour(self, sine_a4):
        clip = load_audio(sine_a4)
        contour = extract_pitch(clip, model="tiny")
        assert isinstance(contour, PitchContour)

    def test_all_arrays_same_length(self, sine_a4):
        clip = load_audio(sine_a4)
        contour = extract_pitch(clip, model="tiny")
        n = len(contour.times)
        assert len(contour.frequencies_hz) == n
        assert len(contour.confidence) == n

    def test_step_size_preserved(self, sine_a4):
        clip = load_audio(sine_a4)
        contour = extract_pitch(clip, model="tiny", step_size_ms=10)
        assert contour.step_size_ms == 10

    def test_model_preserved(self, sine_a4):
        clip = load_audio(sine_a4)
        contour = extract_pitch(clip, model="tiny")
        assert contour.model == "tiny"

    def test_times_monotonically_increasing(self, sine_a4):
        clip = load_audio(sine_a4)
        contour = extract_pitch(clip, model="tiny")
        assert np.all(np.diff(contour.times) > 0)

    def test_confidence_in_unit_interval(self, sine_a4):
        clip = load_audio(sine_a4)
        contour = extract_pitch(clip, model="tiny")
        assert np.all(contour.confidence >= 0.0)
        assert np.all(contour.confidence <= 1.0)


# ----- Output correctness -----

class TestPitchAccuracy:

    def test_a4_sine_predicts_440hz(self, sine_a4):
        """The median predicted pitch on a 440Hz sine should be ~440Hz."""
        clip = load_audio(sine_a4)
        contour = extract_pitch(clip, model="tiny")
        # Use only high-confidence frames; trim the first/last 200ms to avoid
        # edge effects from CREPE's framing.
        voiced = contour.voiced_mask(threshold=0.8)
        mid_mask = (contour.times > 0.2) & (contour.times < 1.3)
        good = voiced & mid_mask
        assert good.sum() > 10, "Not enough confident frames to evaluate"
        median_hz = float(np.median(contour.frequencies_hz[good]))
        # ±10 Hz at 440Hz is roughly ±40 cents — well within tiny-model tolerance.
        assert abs(median_hz - 440.0) < 10.0, f"Expected ~440Hz, got {median_hz}"

    def test_c4_sine_predicts_261hz(self, sine_c4):
        clip = load_audio(sine_c4)
        contour = extract_pitch(clip, model="tiny")
        voiced = contour.voiced_mask(threshold=0.8)
        mid_mask = (contour.times > 0.2) & (contour.times < 1.3)
        good = voiced & mid_mask
        assert good.sum() > 10
        median_hz = float(np.median(contour.frequencies_hz[good]))
        assert abs(median_hz - 261.63) < 10.0, f"Expected ~262Hz, got {median_hz}"


# ----- Error paths -----

class TestPitchErrors:

    def test_wrong_sample_rate_raises(self, sine_a4):
        # Force a non-CREPE rate by loading at 22050.
        clip = load_audio(sine_a4, sample_rate=22050)
        with pytest.raises(PitchExtractionError, match="16000"):
            extract_pitch(clip)


# ----- Voicing mask -----

class TestVoicedMask:

    def test_mask_is_boolean(self, sine_a4):
        clip = load_audio(sine_a4)
        contour = extract_pitch(clip, model="tiny")
        mask = contour.voiced_mask(threshold=0.5)
        assert mask.dtype == np.bool_

    def test_mask_length_matches_contour(self, sine_a4):
        clip = load_audio(sine_a4)
        contour = extract_pitch(clip, model="tiny")
        mask = contour.voiced_mask(threshold=0.5)
        assert len(mask) == contour.num_frames

    def test_higher_threshold_yields_fewer_voiced(self, sine_a4):
        clip = load_audio(sine_a4)
        contour = extract_pitch(clip, model="tiny")
        loose = contour.voiced_mask(threshold=0.3).sum()
        strict = contour.voiced_mask(threshold=0.9).sum()
        assert strict <= loose

    def test_invalid_threshold_raises(self, sine_a4):
        clip = load_audio(sine_a4)
        contour = extract_pitch(clip, model="tiny")
        with pytest.raises(ValueError):
            contour.voiced_mask(threshold=1.5)
        with pytest.raises(ValueError):
            contour.voiced_mask(threshold=-0.1)