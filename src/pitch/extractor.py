"""
Pitch extraction using CREPE.

CREPE (Kim et al., ICASSP 2018) is a deep CNN trained on synthesized and real
audio for monophonic pitch tracking. It outputs a fundamental frequency
estimate and a confidence score per frame, where confidence reflects the
model's certainty that the frame is voiced and the prediction is reliable.

CREPE was trained at 16 kHz. The loader resamples to that rate; this module
sanity-checks the assumption rather than re-resampling, which would be
silent and wasteful.

Model size tradeoffs (config.yaml `pitch.crepe_model`):
    tiny    ~ 1.4 MB,  fastest, ~10% worse RPA than full
    small   ~ 5.4 MB
    medium  ~ 21 MB
    large   ~ 85 MB
    full    ~ 137 MB, slowest, best accuracy
For development we default to `tiny` (already loaded as part of crepe package
import; no network call needed). For final evaluation runs the identifier
will use `full` for reference-quality pitch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import crepe
import numpy as np

from src.audio.loader import AudioClip

# CREPE was trained at this rate; it's not configurable without retraining.
CREPE_SAMPLE_RATE: int = 16000

# Available CREPE model capacities.
CrepeModel = Literal["tiny", "small", "medium", "large", "full"]

# Default analysis hop in milliseconds. CREPE accepts step_size in ms; 10 ms is
# the value used in the paper's evaluation and matches most MIR conventions.
DEFAULT_STEP_SIZE_MS: int = 10


@dataclass(frozen=True)
class PitchContour:
    """
    Frame-by-frame pitch estimates with confidence.

    Fields:
        times: shape (N,), seconds from start of clip for each frame.
        frequencies_hz: shape (N,), pitch in Hz at each frame. Unvoiced frames
            still have a value here (CREPE always emits one); use `confidence`
            or `voiced` to mask.
        confidence: shape (N,), in [0, 1], CREPE's confidence per frame.
        step_size_ms: hop between frames in milliseconds.
        model: which CREPE capacity produced this contour.

    Frozen because downstream code treats pitch as read-only. Transformations
    (cents conversion, smoothing, voicing) produce new arrays rather than
    mutating these.
    """
    times: np.ndarray
    frequencies_hz: np.ndarray
    confidence: np.ndarray
    step_size_ms: int
    model: CrepeModel

    @property
    def num_frames(self) -> int:
        return len(self.times)

    def voiced_mask(self, threshold: float) -> np.ndarray:
        """
        Boolean mask of frames where confidence >= threshold.

        Args:
            threshold: confidence cutoff in [0, 1]. Higher = stricter voicing.

        Returns:
            shape (N,) bool array; True where the frame should be trusted.
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        return self.confidence >= threshold


class PitchExtractionError(RuntimeError):
    """Raised when pitch extraction fails or input violates assumptions."""
    pass


def extract_pitch(
    clip: AudioClip,
    model: CrepeModel = "tiny",
    step_size_ms: int = DEFAULT_STEP_SIZE_MS,
    viterbi: bool = True,
) -> PitchContour:
    """
    Run CREPE on an AudioClip to produce a per-frame pitch contour.

    Args:
        clip: Audio at 16 kHz mono float32 (from `load_audio`).
        model: CREPE model capacity. Smaller = faster, larger = more accurate.
        step_size_ms: Hop between frames in milliseconds.
        viterbi: If True, apply Viterbi smoothing to the pitch path. The paper
            shows this reduces octave errors substantially with negligible
            cost. Recommended on; only turn off for ablations.

    Returns:
        PitchContour with N = ceil(duration_s * 1000 / step_size_ms) frames.

    Raises:
        PitchExtractionError: If `clip.sample_rate` is not 16 kHz (CREPE's
            trained rate) or if CREPE itself fails on the input.
    """
    if clip.sample_rate != CREPE_SAMPLE_RATE:
        raise PitchExtractionError(
            f"CREPE requires {CREPE_SAMPLE_RATE} Hz audio; clip is "
            f"{clip.sample_rate} Hz. Re-load with load_audio(..., sample_rate"
            f"={CREPE_SAMPLE_RATE})."
        )

    try:
        time, frequency, confidence, _activation = crepe.predict(
            clip.samples,
            clip.sample_rate,
            model_capacity=model,
            step_size=step_size_ms,
            viterbi=viterbi,
            verbose=0,
        )
    except Exception as e:
        raise PitchExtractionError(
            f"CREPE failed on {clip.source_path}: {e}"
        ) from e

    return PitchContour(
        times=np.asarray(time, dtype=np.float64),
        frequencies_hz=np.asarray(frequency, dtype=np.float64),
        confidence=np.asarray(confidence, dtype=np.float64),
        step_size_ms=step_size_ms,
        model=model,
    )