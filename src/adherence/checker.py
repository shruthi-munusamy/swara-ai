"""
Ragam adherence checking.

Takes a sequence of swara segments and a target ragam, classifies each
segment as in-raga or out-of-raga, and computes an overall adherence score.

The check is rule-based by design. A learned adherence head would be the
wrong tool here because:
    1. Ragam swara sets are canonical Carnatic theory, not empirical.
       Training a model would risk overfitting to dataset artifacts
       rather than the underlying rules.
    2. The transformation we want (cents-fold pitch → membership check)
       is closed-form. Any model would just be reproducing this with
       extra noise.
    3. The interesting "this region is out of raga" claim needs to be
       directly explainable to a user (which swara, which time, why it's
       wrong). Rule-based output gives this for free.

Disambiguation between shared-cents-position swaras (R2/G1, R3/G2, D2/N1,
D3/N2) happens here: the segmenter outputs both candidate names because
it has no ragam context, and the checker picks whichever is in the target
ragam's swara set. If neither is in the set, the segment is out-of-raga.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from src.swara.definitions import swara_to_cents
from src.swara.ragams import Ragam
from src.swara.segmenter import SwaraSegment


@dataclass(frozen=True)
class AdherenceSegment:
    """
    A swara segment with its in-raga/out-of-raga classification.

    Fields:
        segment: the source SwaraSegment.
        in_raga: True if at least one candidate swara is in the target ragam.
        matched_swara: the candidate name that's in the ragam (if in_raga),
            chosen by intersecting the segment's candidates with the ragam's
            allowed swaras. None when out-of-raga.
        deviation_cents: signed distance from the matched swara's exact
            cents position, in cents. Positive = sharp, negative = flat.
            None when out-of-raga.
    """
    segment: SwaraSegment
    in_raga: bool
    matched_swara: Optional[str]
    deviation_cents: Optional[float]


@dataclass(frozen=True)
class AdherenceReport:
    """
    Summary of how well a sequence of segments adheres to a target ragam.

    Fields:
        ragam: the target ragam used for the check.
        per_segment: classification of each segment, in time order.
        adherence_score: confidence-weighted fraction of segment duration
            that's in-raga, in [0, 1]. Computed as
                sum(duration * confidence  for in-raga segments)
                /
                sum(duration * confidence  for all scored segments)
            Segments below `min_confidence` are excluded from both sums.
        scored_duration_seconds: total duration that contributed to the
            score, after the confidence filter.
        skipped_duration_seconds: total duration excluded due to low
            confidence. Useful for "I trust this score over N% of the audio."
        out_of_raga_segments: convenience list of segments flagged out-of-raga.
    """
    ragam: Ragam
    per_segment: List[AdherenceSegment]
    adherence_score: float
    scored_duration_seconds: float
    skipped_duration_seconds: float
    out_of_raga_segments: List[AdherenceSegment]

    @property
    def num_segments(self) -> int:
        return len(self.per_segment)

    @property
    def num_out_of_raga(self) -> int:
        return len(self.out_of_raga_segments)


def _classify_segment(segment: SwaraSegment, ragam: Ragam) -> AdherenceSegment:
    """
    Classify a single segment against a target ragam.

    Picks the candidate swara that's in the ragam (if any). When multiple
    candidates are valid (rare: usually only one is in any given ragam),
    picks the one closest to the segment's measured cents.
    """
    in_ragam_candidates = [
        name for name in segment.swara_candidates
        if name in ragam.swaras
    ]

    if not in_ragam_candidates:
        return AdherenceSegment(
            segment=segment,
            in_raga=False,
            matched_swara=None,
            deviation_cents=None,
        )

    if len(in_ragam_candidates) == 1:
        matched = in_ragam_candidates[0]
    else:
        # Multiple ragam-valid candidates: choose the one whose exact cents
        # position is closest to the segment's measured cents. In practice
        # this branch is almost never taken because the disambiguating pairs
        # (R2/G1, R3/G2, D2/N1, D3/N2) by construction occupy the same cents
        # position, and a ragam typically uses one or the other but not both.
        matched = min(
            in_ragam_candidates,
            key=lambda name: abs(swara_to_cents(name) - segment.cents),
        )

    deviation = segment.cents - swara_to_cents(matched)
    return AdherenceSegment(
        segment=segment,
        in_raga=True,
        matched_swara=matched,
        deviation_cents=deviation,
    )


def check_adherence(
    segments: List[SwaraSegment],
    ragam: Ragam,
    min_confidence: float = 0.5,
) -> AdherenceReport:
    """
    Score a sequence of swara segments against a target ragam.

    Args:
        segments: time-ordered list from `segment_swaras`.
        ragam: target ragam from the registry.
        min_confidence: per-segment CREPE confidence cutoff for scoring.
            Segments below this contribute to `skipped_duration_seconds`
            instead of the score. Default 0.5 matches `config.yaml`.

    Returns:
        AdherenceReport with per-segment labels and overall score.

    Raises:
        ValueError: If min_confidence is outside [0, 1].
    """
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError(
            f"min_confidence must be in [0, 1], got {min_confidence}"
        )

    per_segment = [_classify_segment(seg, ragam) for seg in segments]
    out_of_raga = [a for a in per_segment if not a.in_raga]

    scored_duration = 0.0
    in_raga_weighted_duration = 0.0
    skipped_duration = 0.0

    for ann in per_segment:
        seg = ann.segment
        weight = seg.duration_seconds * seg.confidence
        if seg.confidence < min_confidence:
            skipped_duration += seg.duration_seconds
            continue
        scored_duration += seg.duration_seconds
        if ann.in_raga:
            in_raga_weighted_duration += weight

    total_weighted_duration = sum(
        seg.segment.duration_seconds * seg.segment.confidence
        for seg in per_segment
        if seg.segment.confidence >= min_confidence
    )

    if total_weighted_duration > 0:
        adherence_score = in_raga_weighted_duration / total_weighted_duration
    else:
        adherence_score = 0.0

    return AdherenceReport(
        ragam=ragam,
        per_segment=per_segment,
        adherence_score=adherence_score,
        scored_duration_seconds=scored_duration,
        skipped_duration_seconds=skipped_duration,
        out_of_raga_segments=out_of_raga,
    )