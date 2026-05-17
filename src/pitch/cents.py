"""
Tonic-relative cents conversion.

Converts an absolute-Hz pitch contour into cents relative to a singer's
shruthi (tonic). This is the architectural primitive that makes the rest
of the pipeline shruthi-invariant.

The cents formula:
    cents = 1200 * log2(freq_hz / shruthi_hz)

Properties worth knowing:
    - cents is additive under transposition: shifting all pitches by N
      semitones (multiplicatively in Hz) adds 100*N to every cents value.
    - cents is zero at the tonic, positive above, negative below.
    - one octave = 1200 cents; one semitone = 100 cents.

Octave folding (`octave_fold=True`) reduces every cents value to [0, 1200),
which is what the segmenter and adherence checker want: they care about
*which swara* the pitch represents, not which octave it lives in. The
identifier may want unfolded cents to capture register information, so
folding is opt-in rather than automatic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.pitch.extractor import PitchContour

# Plausible shruthi range for human voice, in Hz.
# Male singers typically sing at ~130-180 Hz Sa; female singers ~200-280 Hz.
# Tighter bounds would reject legitimate edge cases (very low male, very high
# female); looser bounds would let typos through (440 Hz Sa is unrealistically
# high). 80-400 Hz is a generous-but-not-silly window.
MIN_SHRUTHI_HZ: float = 80.0
MAX_SHRUTHI_HZ: float = 400.0


@dataclass(frozen=True)
class CentsContour:
    """
    Frame-by-frame pitch in cents relative to a singer's shruthi.

    Fields:
        times: shape (N,), seconds from start of clip for each frame.
        cents: shape (N,), pitch in cents relative to shruthi. Unvoiced
            frames carry junk values; use `voiced` to mask before consuming.
        confidence: shape (N,), CREPE's confidence per frame, in [0, 1].
        voiced: shape (N,), bool mask of frames trusted enough to use.
        shruthi_hz: tonic frequency used for the conversion.
        octave_folded: True if cents values were reduced to [0, 1200).
        step_size_ms: hop between frames in milliseconds (preserved from
            the source PitchContour for downstream timing math).

    Frozen because cents are read-only after conversion. Downstream code
    that wants to filter, smooth, or fold produces new contours.
    """
    times: np.ndarray
    cents: np.ndarray
    confidence: np.ndarray
    voiced: np.ndarray
    shruthi_hz: float
    octave_folded: bool
    step_size_ms: int

    @property
    def num_frames(self) -> int:
        return len(self.times)

    def voiced_cents(self) -> np.ndarray:
        """
        Return only the cents values where `voiced` is True.

        Common access pattern for the segmenter and adherence checker,
        which discard unvoiced frames entirely.
        """
        return self.cents[self.voiced]

    def voiced_times(self) -> np.ndarray:
        """Times corresponding to `voiced_cents()`."""
        return self.times[self.voiced]


class CentsConversionError(ValueError):
    """Raised when shruthi is invalid or input contour is malformed."""
    pass


def hz_to_cents(freq_hz: np.ndarray, shruthi_hz: float) -> np.ndarray:
    """
    Convert an array of frequencies in Hz to cents relative to shruthi.

    Args:
        freq_hz: shape (N,) array of frequencies. Non-positive entries (CREPE
            occasionally produces zero or negative on silence) produce NaN
            rather than -inf to keep downstream masking clean.
        shruthi_hz: tonic frequency in Hz. Must be positive and within plausible
            human vocal range.

    Returns:
        shape (N,) array of cents values. NaN where input was non-positive.

    Raises:
        CentsConversionError: If shruthi is out of plausible range.
    """
    if not MIN_SHRUTHI_HZ <= shruthi_hz <= MAX_SHRUTHI_HZ:
        raise CentsConversionError(
            f"shruthi_hz={shruthi_hz} is outside plausible vocal range "
            f"[{MIN_SHRUTHI_HZ}, {MAX_SHRUTHI_HZ}]. If this is intentional "
            f"(e.g. instrument analysis), call _hz_to_cents_unchecked directly."
        )
    return _hz_to_cents_unchecked(freq_hz, shruthi_hz)


def _hz_to_cents_unchecked(freq_hz: np.ndarray, shruthi_hz: float) -> np.ndarray:
    """Internal helper: cents math with no range check on shruthi."""
    freq_hz = np.asarray(freq_hz, dtype=np.float64)
    # Suppress numpy's divide/log warnings for the non-positive entries we're
    # about to overwrite with NaN.
    with np.errstate(divide="ignore", invalid="ignore"):
        cents = 1200.0 * np.log2(freq_hz / shruthi_hz)
    cents = np.where(freq_hz > 0.0, cents, np.nan)
    return cents


def fold_to_octave(cents: np.ndarray) -> np.ndarray:
    """
    Reduce cents values to the half-open interval [0, 1200).

    NaN values pass through unchanged.

    Args:
        cents: shape (N,) array of cents values, possibly negative or > 1200.

    Returns:
        shape (N,) array with each value mapped to its equivalent
        position within a single octave [0, 1200).
    """
    cents = np.asarray(cents, dtype=np.float64)
    return np.where(np.isnan(cents), cents, np.mod(cents, 1200.0))


def to_cents_contour(
    contour: PitchContour,
    shruthi_hz: float,
    confidence_threshold: float,
    octave_fold: bool = True,
) -> CentsContour:
    """
    Convert a Hz-valued PitchContour into a tonic-relative CentsContour.

    Args:
        contour: PitchContour from `extract_pitch`.
        shruthi_hz: singer's tonic in Hz.
        confidence_threshold: voicing cutoff applied via `contour.voiced_mask`.
            Frames below this are still included in the output (with junk
            cents values), but flagged as unvoiced.
        octave_fold: if True (default), fold cents to [0, 1200). The segmenter
            and adherence checker expect folded values; the identifier may
            want unfolded for register information.

    Returns:
        A CentsContour with cents, voicing, and timing aligned frame-by-frame
        with the source PitchContour.

    Raises:
        CentsConversionError: If shruthi is out of plausible vocal range, or
            if the contour is empty.
    """
    if contour.num_frames == 0:
        raise CentsConversionError("Cannot convert an empty PitchContour")

    cents = hz_to_cents(contour.frequencies_hz, shruthi_hz)
    if octave_fold:
        cents = fold_to_octave(cents)

    voiced = contour.voiced_mask(confidence_threshold)

    return CentsContour(
        times=contour.times,
        cents=cents,
        confidence=contour.confidence,
        voiced=voiced,
        shruthi_hz=shruthi_hz,
        octave_folded=octave_fold,
        step_size_ms=contour.step_size_ms,
    )