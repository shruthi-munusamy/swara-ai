"""Tests for the audio loader."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.audio.loader import (
    DEFAULT_SAMPLE_RATE,
    MIN_DURATION_SECONDS,
    AudioClip,
    AudioLoadError,
    load_audio,
    peak_normalize,
)


# ----- Synthetic audio fixtures -----

def _write_sine(
    path: Path,
    freq_hz: float = 440.0,
    duration_s: float = 1.0,
    sample_rate: int = 44100,
    amplitude: float = 0.5,
    channels: int = 1,
) -> Path:
    """Write a sine wave to disk. Used to construct test fixtures."""
    n = int(duration_s * sample_rate)
    t = np.arange(n) / sample_rate
    wave = amplitude * np.sin(2.0 * np.pi * freq_hz * t).astype(np.float32)
    if channels == 2:
        wave = np.stack([wave, wave], axis=-1)
    sf.write(str(path), wave, sample_rate)
    return path


@pytest.fixture
def sine_wav(tmp_path) -> Path:
    """A 1-second 440 Hz mono sine at 44.1 kHz."""
    return _write_sine(tmp_path / "sine.wav")


@pytest.fixture
def stereo_wav(tmp_path) -> Path:
    """A 1-second 440 Hz stereo sine at 44.1 kHz."""
    return _write_sine(tmp_path / "stereo.wav", channels=2)


@pytest.fixture
def silent_wav(tmp_path) -> Path:
    """A 1-second silent mono WAV."""
    path = tmp_path / "silent.wav"
    sf.write(str(path), np.zeros(44100, dtype=np.float32), 44100)
    return path


@pytest.fixture
def too_short_wav(tmp_path) -> Path:
    """A 50 ms WAV — below MIN_DURATION_SECONDS."""
    return _write_sine(tmp_path / "short.wav", duration_s=0.05)


# ----- Loading basics -----

class TestLoadAudio:

    def test_returns_audio_clip(self, sine_wav):
        clip = load_audio(sine_wav)
        assert isinstance(clip, AudioClip)

    def test_resamples_to_default_rate(self, sine_wav):
        clip = load_audio(sine_wav)
        assert clip.sample_rate == DEFAULT_SAMPLE_RATE

    def test_samples_are_mono(self, stereo_wav):
        clip = load_audio(stereo_wav)
        assert clip.samples.ndim == 1

    def test_samples_are_float32(self, sine_wav):
        clip = load_audio(sine_wav)
        assert clip.samples.dtype == np.float32

    def test_samples_in_valid_range(self, sine_wav):
        clip = load_audio(sine_wav)
        assert np.max(np.abs(clip.samples)) <= 1.0

    def test_duration_correct(self, sine_wav):
        clip = load_audio(sine_wav)
        # 1-second source clip, allow tiny tolerance for resampling edge effects.
        assert abs(clip.duration_seconds - 1.0) < 0.01

    def test_source_path_preserved(self, sine_wav):
        clip = load_audio(sine_wav)
        assert clip.source_path == sine_wav

    def test_custom_sample_rate(self, sine_wav):
        clip = load_audio(sine_wav, sample_rate=22050)
        assert clip.sample_rate == 22050


# ----- Error paths -----

class TestLoadAudioErrors:

    def test_nonexistent_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_audio(tmp_path / "does_not_exist.wav")

    def test_unsupported_extension_raises(self, tmp_path):
        bad = tmp_path / "audio.xyz"
        bad.write_bytes(b"not audio")
        with pytest.raises(AudioLoadError, match="Unsupported"):
            load_audio(bad)

    def test_too_short_audio_raises(self, too_short_wav):
        with pytest.raises(AudioLoadError, match="minimum"):
            load_audio(too_short_wav)


# ----- AudioClip properties -----

class TestAudioClipProperties:

    def test_duration_seconds(self, sine_wav):
        clip = load_audio(sine_wav)
        assert clip.duration_seconds == len(clip.samples) / clip.sample_rate

    def test_num_samples(self, sine_wav):
        clip = load_audio(sine_wav)
        assert clip.num_samples == len(clip.samples)


# ----- Peak normalization -----

class TestPeakNormalize:

    def test_scales_to_target_peak(self, sine_wav):
        clip = load_audio(sine_wav)
        normalized = peak_normalize(clip, target_peak=0.95)
        assert abs(np.max(np.abs(normalized.samples)) - 0.95) < 1e-5

    def test_returns_new_clip(self, sine_wav):
        clip = load_audio(sine_wav)
        normalized = peak_normalize(clip)
        assert normalized is not clip

    def test_preserves_sample_rate(self, sine_wav):
        clip = load_audio(sine_wav)
        normalized = peak_normalize(clip)
        assert normalized.sample_rate == clip.sample_rate

    def test_preserves_source_path(self, sine_wav):
        clip = load_audio(sine_wav)
        normalized = peak_normalize(clip)
        assert normalized.source_path == clip.source_path

    def test_silent_audio_raises(self, silent_wav):
        clip = load_audio(silent_wav)
        with pytest.raises(ValueError, match="silent"):
            peak_normalize(clip)

    def test_invalid_target_peak_raises(self, sine_wav):
        clip = load_audio(sine_wav)
        with pytest.raises(ValueError, match="target_peak"):
            peak_normalize(clip, target_peak=1.5)
        with pytest.raises(ValueError, match="target_peak"):
            peak_normalize(clip, target_peak=0.0)