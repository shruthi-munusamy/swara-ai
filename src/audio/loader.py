"""
Audio loading and preprocessing.

Single entry point for reading audio files into the pipeline. All downstream
components consume the AudioClip dataclass produced here, which guarantees:
    - mono float32 samples in [-1.0, 1.0]
    - sample rate matching `config.audio.sample_rate` (16 kHz for CREPE)
    - duration available without re-reading the file

Centralizing the loader prevents the "every module loads audio slightly
differently" failure mode where one component normalizes peaks and another
doesn't, or sample rates silently mismatch and pitch values shift by a
semitone across the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import librosa
import numpy as np

# Pinned defaults. Override per-call only if you have a specific reason.
DEFAULT_SAMPLE_RATE: int = 16000
SUPPORTED_EXTENSIONS = frozenset({".wav", ".mp3", ".flac", ".ogg", ".m4a"})

# Minimum duration we'll accept as a meaningful audio clip. Anything shorter
# is almost certainly an error (empty file, corrupt header, accidental click).
MIN_DURATION_SECONDS: float = 0.1


@dataclass(frozen=True)
class AudioClip:
    """
    A loaded audio clip with mono float32 samples at a known sample rate.

    Frozen because once loaded, audio is treated as read-only by the rest of
    the pipeline. Transformations (pitch extraction, segmentation, etc.)
    produce new objects rather than mutating this one.

    Fields:
        samples: 1-D float32 numpy array, values in [-1.0, 1.0], mono.
        sample_rate: samples per second.
        source_path: path on disk this was loaded from (for error messages
            and downstream metadata; not used in any computation).
    """
    samples: np.ndarray
    sample_rate: int
    source_path: Path

    @property
    def duration_seconds(self) -> float:
        """Length of the clip in seconds."""
        return len(self.samples) / self.sample_rate

    @property
    def num_samples(self) -> int:
        """Total number of audio samples."""
        return len(self.samples)


class AudioLoadError(ValueError):
    """Raised when audio fails to load or violates expected properties."""
    pass


def load_audio(
    path: Union[str, Path],
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> AudioClip:
    """
    Load an audio file, resampled to mono at the target sample rate.

    Args:
        path: Path to the audio file. Supported: WAV, MP3, FLAC, OGG, M4A.
        sample_rate: Target sample rate in Hz. Defaults to 16000 (CREPE's
            expected input rate). Override only for non-CREPE pipelines.

    Returns:
        AudioClip with mono float32 samples at `sample_rate`.

    Raises:
        FileNotFoundError: If `path` does not exist.
        AudioLoadError: If the extension is unsupported, the file fails to
            decode, or the resulting clip is shorter than MIN_DURATION_SECONDS.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise AudioLoadError(
            f"Unsupported audio extension {path.suffix!r}. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    try:
        # mono=True averages multi-channel sources; sr=sample_rate resamples
        # using soxr (high-quality kaiser_best is librosa's default).
        # dtype=np.float32 keeps memory usage and downstream tensor conversion cheap.
        samples, loaded_sr = librosa.load(
            str(path),
            sr=sample_rate,
            mono=True,
            dtype=np.float32,
        )
    except Exception as e:
        raise AudioLoadError(f"Failed to load {path}: {e}") from e

    # Sanity check: librosa should always return the requested SR; this catches
    # the case where it falls back silently due to a backend issue.
    if loaded_sr != sample_rate:
        raise AudioLoadError(
            f"Expected sample rate {sample_rate}, got {loaded_sr} from {path}"
        )

    clip = AudioClip(
        samples=samples,
        sample_rate=loaded_sr,
        source_path=path,
    )

    if clip.duration_seconds < MIN_DURATION_SECONDS:
        raise AudioLoadError(
            f"Audio at {path} is {clip.duration_seconds:.3f}s; "
            f"minimum is {MIN_DURATION_SECONDS}s. Likely a corrupt or empty file."
        )

    return clip


def peak_normalize(clip: AudioClip, target_peak: float = 0.95) -> AudioClip:
    """
    Return a new AudioClip with samples scaled so the absolute peak equals
    `target_peak`. Useful for normalizing input level before pitch tracking
    on quiet recordings; not applied by default because CREPE is robust to
    input level and unnecessary scaling can mask actual recording issues.

    Args:
        clip: Audio to normalize.
        target_peak: Desired peak absolute amplitude. Must be in (0.0, 1.0].

    Returns:
        New AudioClip with samples scaled.

    Raises:
        ValueError: If target_peak is out of range or the clip is silent
            (peak amplitude is zero).
    """
    if not 0.0 < target_peak <= 1.0:
        raise ValueError(f"target_peak must be in (0.0, 1.0], got {target_peak}")

    current_peak = float(np.max(np.abs(clip.samples)))
    if current_peak == 0.0:
        raise ValueError("Cannot normalize silent audio (peak amplitude is zero)")

    scaled = clip.samples * (target_peak / current_peak)
    return AudioClip(
        samples=scaled.astype(np.float32),
        sample_rate=clip.sample_rate,
        source_path=clip.source_path,
    )