"""
Swara segmentation from a tonic-relative cents contour.

Converts a frame-by-frame CentsContour into a sequence of SwaraSegment
objects, each representing a stable region of pitch identified as a
specific swarasthana.

The pipeline applied here:
    1. Smooth the cents contour with a median filter to suppress
       short-period gamaka oscillations and pitch tracker jitter.
    2. For each frame, find the nearest swarasthana within
       `tolerance_cents`. Frames with no swara within tolerance are
       labeled as transitions and excluded from segments.
    3. Run-length-encode the swara labels into contiguous segments.
    4. Drop segments shorter than `min_stable_ms` — too brief to count.

What this module deliberately doesn't do:
    - Detect or characterize gamakas (kampita, jaru, spurita). A long
      oscillation around a target swara gets fragmented here; identifying
      it as one gamaka event is the job of a future GamakaDetector built
      on top of these segments.
    - Distinguish R2 from G1 (or R3/G2, D2/N1, D3/N2). These share cents
      values; the segmenter outputs both candidate names and ragam context
      (in the adherence checker) is what picks one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy.signal import medfilt

from src.pitch.cents import CentsContour
from src.swara.definitions import SWARASTHANA_CENTS


@dataclass(frozen=True)
class SwaraSegment:
    """
    A contiguous stable region of pitch labeled with a swarasthana.

    Fields:
        start_time: seconds from start of clip.
        end_time: seconds from start of clip.
        swara_candidates: list of swarasthana names that this segment's
            cents value is closest to. Length > 1 when multiple swaras
            share the same cents position (e.g. R2 and G1 both at 200c);
            adherence/identification logic disambiguates using ragam context.
        cents: the segment's representative cents value (median of frames).
        confidence: mean CREPE confidence across the segment's frames.
        num_frames: how many source frames contributed.

    Frozen because segments are produced once per audio clip and consumed
    downstream as read-only labels.
    """
    start_time: float
    end_time: float
    swara_candidates: Tuple[str, ...]
    cents: float
    confidence: float
    num_frames: int

    @property
    def duration_seconds(self) -> float:
        return self.end_time - self.start_time

    @property
    def primary_candidate(self) -> str:
        """
        Return the alphabetically-first candidate name. Useful for display
        when ragam context isn't available. Real disambiguation happens in
        the adherence checker.
        """
        return self.swara_candidates[0]


class SegmentationError(ValueError):
    """Raised when segmentation inputs violate assumptions."""
    pass


def _build_swara_index() -> Tuple[np.ndarray, List[Tuple[str, ...]]]:
    """
    Build lookup structures mapping cents positions to swarasthana names.

    Multiple swaras can occupy the same cents position (R2/G1 at 200c, etc.).
    The index groups by cents so a single position returns all valid names.

    Returns:
        cents_positions: shape (K,) sorted array of unique cents values.
        names_per_position: list of K tuples, each tuple containing the
            swara names sharing that cents value.
    """
    cents_to_names: dict[float, List[str]] = {}
    for name, cents in SWARASTHANA_CENTS.items():
        cents_to_names.setdefault(cents, []).append(name)

    sorted_cents = sorted(cents_to_names.keys())
    cents_positions = np.asarray(sorted_cents, dtype=np.float64)
    names_per_position = [
        tuple(sorted(cents_to_names[c])) for c in sorted_cents
    ]
    return cents_positions, names_per_position


# Built once at module import; the swarasthana table is constant.
_CENTS_POSITIONS, _NAMES_PER_POSITION = _build_swara_index()


def _smooth_cents(cents: np.ndarray, window_frames: int) -> np.ndarray:
    """
    Apply a median filter to the cents contour.

    Median (rather than mean) suppresses brief gamaka excursions without
    smearing them into adjacent stable regions. NaN frames are preserved.

    Args:
        cents: shape (N,) cents values, possibly containing NaN.
        window_frames: median filter window. Must be odd and >= 1.

    Returns:
        shape (N,) smoothed array. NaN frames in the input remain NaN.

    Raises:
        SegmentationError: if window_frames is invalid.
    """
    if window_frames < 1:
        raise SegmentationError(f"window_frames must be >= 1, got {window_frames}")
    if window_frames % 2 == 0:
        raise SegmentationError(
            f"window_frames must be odd, got {window_frames}"
        )
    if window_frames == 1:
        return cents.copy()

    # scipy's medfilt doesn't handle NaN naturally — it propagates them
    # aggressively, which makes voiced regions adjacent to unvoiced gaps
    # disappear. We use a sentinel value during filtering, then restore
    # NaN at the original positions.
    nan_mask = np.isnan(cents)
    safe = np.where(nan_mask, 0.0, cents)
    smoothed = medfilt(safe, kernel_size=window_frames)
    return np.where(nan_mask, np.nan, smoothed)


def _quantize_to_swara(
    cents_value: float,
    tolerance_cents: float,
) -> Tuple[str, ...] | None:
    """
    Find the swarasthana(s) within tolerance of a cents value.

    Returns:
        Tuple of swara names if a position is within tolerance, else None.
        Multiple names returned when the position is shared (R2/G1, etc.).
    """
    if np.isnan(cents_value):
        return None
    distances = np.abs(_CENTS_POSITIONS - cents_value)
    nearest = int(np.argmin(distances))
    if distances[nearest] > tolerance_cents:
        return None
    return _NAMES_PER_POSITION[nearest]


def segment_swaras(
    contour: CentsContour,
    tolerance_cents: float,
    min_stable_ms: float,
    smoothing_window: int,
) -> List[SwaraSegment]:
    """
    Segment a CentsContour into stable swara regions.

    Args:
        contour: tonic-relative cents contour, octave-folded. Must have
            `octave_folded=True` because the swarasthana table lives in
            [0, 1200) and unfolded cents would never match.
        tolerance_cents: a frame matches a swarasthana if its smoothed cents
            value is within this distance of the swarasthana's cents
            position. Same units (cents) as the cents axis.
        min_stable_ms: discard segments shorter than this. Below ~50ms is
            transient; below ~30ms is almost certainly noise.
        smoothing_window: median filter window in frames. Odd, >= 1.

    Returns:
        List of SwaraSegment in time order. Empty if no segments meet the
        duration threshold (silent clip, all unvoiced, all transitions).

    Raises:
        SegmentationError: if the contour is not octave-folded, or if
            parameter values are invalid.
    """
    if not contour.octave_folded:
        raise SegmentationError(
            "segment_swaras requires an octave-folded CentsContour. "
            "Call to_cents_contour(..., octave_fold=True) upstream."
        )
    if tolerance_cents <= 0:
        raise SegmentationError(
            f"tolerance_cents must be positive, got {tolerance_cents}"
        )
    if min_stable_ms <= 0:
        raise SegmentationError(
            f"min_stable_ms must be positive, got {min_stable_ms}"
        )

    n = contour.num_frames
    if n == 0:
        return []

    # Apply voicing first: unvoiced frames are NaN'd so they can't form segments.
    masked_cents = np.where(contour.voiced, contour.cents, np.nan)
    smoothed = _smooth_cents(masked_cents, smoothing_window)

    # Label each frame with its swara candidate tuple (or None for transitions).
    frame_labels: List[Tuple[str, ...] | None] = [
        _quantize_to_swara(c, tolerance_cents) for c in smoothed
    ]

    # Run-length-encode contiguous identical labels.
    segments: List[SwaraSegment] = []
    min_stable_seconds = min_stable_ms / 1000.0

    i = 0
    while i < n:
        label = frame_labels[i]
        if label is None:
            i += 1
            continue

        # Walk forward while the label stays the same.
        j = i
        while j < n and frame_labels[j] == label:
            j += 1
        # Run is [i, j); frames i..j-1 share the same swara label.

        start_time = float(contour.times[i])
        # Use the *next* frame's time (or contour end) as the segment end,
        # so end_time - start_time accurately reflects how long the swara
        # was held, including the duration of its last contributing frame.
        if j < n:
            end_time = float(contour.times[j])
        else:
            end_time = float(contour.times[j - 1]) + contour.step_size_ms / 1000.0

        duration = end_time - start_time
        if duration >= min_stable_seconds:
            segment_cents = float(np.nanmedian(smoothed[i:j]))
            segment_confidence = float(np.nanmean(contour.confidence[i:j]))
            segments.append(
                SwaraSegment(
                    start_time=start_time,
                    end_time=end_time,
                    swara_candidates=label,
                    cents=segment_cents,
                    confidence=segment_confidence,
                    num_frames=j - i,
                )
            )

        i = j

    return segments