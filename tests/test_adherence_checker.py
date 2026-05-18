"""Tests for ragam adherence checking."""

import pytest

from src.adherence.checker import (
    AdherenceReport,
    AdherenceSegment,
    check_adherence,
)
from src.swara.definitions import swara_to_cents
from src.swara.ragams import load_ragams
from src.swara.segmenter import SwaraSegment


@pytest.fixture(scope="module")
def registry():
    return load_ragams()


@pytest.fixture(scope="module")
def sankarabharanam(registry):
    """Mela 29 — S R2 G3 M1 P D2 N3."""
    return registry.get_by_number(29)


@pytest.fixture(scope="module")
def kalyani(registry):
    """Mela 65 — S R2 G3 M2 P D2 N3. Differs from Sankarabharanam by M2 vs M1."""
    return registry.get_by_number(65)


def _make_segment(
    candidates: tuple,
    cents: float,
    start: float = 0.0,
    end: float = 0.5,
    confidence: float = 0.9,
) -> SwaraSegment:
    return SwaraSegment(
        start_time=start,
        end_time=end,
        swara_candidates=candidates,
        cents=cents,
        confidence=confidence,
        num_frames=int((end - start) * 100),
    )


# ----- Basic in-raga / out-of-raga classification -----

class TestClassification:

    def test_sa_is_in_every_ragam(self, sankarabharanam):
        seg = _make_segment(("S",), cents=swara_to_cents("S"))
        report = check_adherence([seg], sankarabharanam)
        assert report.per_segment[0].in_raga
        assert report.per_segment[0].matched_swara == "S"

    def test_r2_in_sankarabharanam(self, sankarabharanam):
        seg = _make_segment(("R2", "G1"), cents=swara_to_cents("R2"))
        report = check_adherence([seg], sankarabharanam)
        # Sankarabharanam uses R2 but not G1 — checker should pick R2.
        assert report.per_segment[0].in_raga
        assert report.per_segment[0].matched_swara == "R2"

    def test_g1_not_in_sankarabharanam(self, sankarabharanam):
        """G1 (200c) shares position with R2 but isn't named in Sankarabharanam.
        The checker should still classify this as in-raga via R2."""
        seg = _make_segment(("G1",), cents=swara_to_cents("G1"))
        report = check_adherence([seg], sankarabharanam)
        # Segment had only G1 as a candidate; G1 is not in Sankarabharanam's
        # swara set. So it's out of raga.
        assert not report.per_segment[0].in_raga

    def test_m2_not_in_sankarabharanam(self, sankarabharanam):
        """M2 is Kalyani's defining swara, foreign to Sankarabharanam."""
        seg = _make_segment(("M2",), cents=swara_to_cents("M2"))
        report = check_adherence([seg], sankarabharanam)
        assert not report.per_segment[0].in_raga
        assert report.per_segment[0].matched_swara is None

    def test_m1_in_sankarabharanam_not_kalyani(self, sankarabharanam, kalyani):
        """The distinguishing test: M1 is in 29, M2 is in 65."""
        m1 = _make_segment(("M1",), cents=swara_to_cents("M1"))
        m2 = _make_segment(("M2",), cents=swara_to_cents("M2"))

        report_sankara = check_adherence([m1, m2], sankarabharanam)
        report_kalyani = check_adherence([m1, m2], kalyani)

        # In Sankarabharanam: M1 passes, M2 fails.
        assert report_sankara.per_segment[0].in_raga
        assert not report_sankara.per_segment[1].in_raga

        # In Kalyani: M1 fails, M2 passes.
        assert not report_kalyani.per_segment[0].in_raga
        assert report_kalyani.per_segment[1].in_raga


# ----- Deviation reporting -----

class TestDeviation:

    def test_exact_pitch_has_zero_deviation(self, sankarabharanam):
        seg = _make_segment(("R2", "G1"), cents=swara_to_cents("R2"))
        report = check_adherence([seg], sankarabharanam)
        assert report.per_segment[0].deviation_cents == pytest.approx(0.0)

    def test_sharp_deviation_positive(self, sankarabharanam):
        seg = _make_segment(("S",), cents=15.0)  # S = 0, sung sharp by 15 cents
        report = check_adherence([seg], sankarabharanam)
        assert report.per_segment[0].deviation_cents == pytest.approx(15.0)

    def test_flat_deviation_negative(self, sankarabharanam):
        # Sankarabharanam uses N3 at 1100c; if the singer hits 1085c, that's
        # 15 cents flat.
        seg = _make_segment(("N3",), cents=1085.0)
        report = check_adherence([seg], sankarabharanam)
        assert report.per_segment[0].deviation_cents == pytest.approx(-15.0)

    def test_out_of_raga_has_no_deviation(self, sankarabharanam):
        seg = _make_segment(("M2",), cents=swara_to_cents("M2"))
        report = check_adherence([seg], sankarabharanam)
        assert report.per_segment[0].deviation_cents is None


# ----- Adherence score -----

class TestAdherenceScore:

    def test_all_in_raga_scores_one(self, sankarabharanam):
        segs = [
            _make_segment(("S",), cents=0.0),
            _make_segment(("R2",), cents=200.0),
            _make_segment(("G3",), cents=400.0),
            _make_segment(("P",), cents=700.0),
        ]
        report = check_adherence(segs, sankarabharanam)
        assert report.adherence_score == pytest.approx(1.0)

    def test_all_out_of_raga_scores_zero(self, sankarabharanam):
        segs = [
            _make_segment(("R1",), cents=100.0),  # Sankarabharanam has R2, not R1
            _make_segment(("G2",), cents=300.0),  # Sankarabharanam has G3
            _make_segment(("M2",), cents=600.0),  # Sankarabharanam has M1
        ]
        report = check_adherence(segs, sankarabharanam)
        assert report.adherence_score == pytest.approx(0.0)

    def test_mixed_yields_intermediate(self, sankarabharanam):
        """Equal-duration mix: half in, half out → score around 0.5."""
        segs = [
            _make_segment(("S",), cents=0.0, start=0.0, end=0.5),
            _make_segment(("M2",), cents=600.0, start=0.5, end=1.0),  # foreign
        ]
        report = check_adherence(segs, sankarabharanam)
        assert report.adherence_score == pytest.approx(0.5, abs=0.01)

    def test_duration_weighted(self, sankarabharanam):
        """Long in-raga segment + short out-of-raga should score high."""
        segs = [
            _make_segment(("S",), cents=0.0, start=0.0, end=4.0),    # 4s in
            _make_segment(("M2",), cents=600.0, start=4.0, end=5.0),  # 1s out
        ]
        report = check_adherence(segs, sankarabharanam)
        # 4 of 5 weighted seconds are in-raga; confidence is uniform.
        assert report.adherence_score == pytest.approx(0.8, abs=0.01)

    def test_confidence_weighted(self, sankarabharanam):
        """An in-raga segment with low confidence weighs less than a
        confident one of the same duration."""
        segs = [
            _make_segment(("S",), cents=0.0, confidence=0.9),       # in, weight 0.45
            _make_segment(("M2",), cents=600.0, confidence=0.55),   # out, weight 0.275
        ]
        report = check_adherence(segs, sankarabharanam)
        # in-raga weighted: 0.5 * 0.9 = 0.45; total weighted: 0.45 + 0.275 = 0.725
        assert report.adherence_score == pytest.approx(0.45 / 0.725, abs=0.01)


# ----- Confidence filtering -----

class TestConfidenceFiltering:

    def test_low_confidence_segments_excluded_from_score(self, sankarabharanam):
        """A foreign swara below the confidence threshold shouldn't drag
        the score down — it's excluded entirely."""
        segs = [
            _make_segment(("S",), cents=0.0, confidence=0.9),
            _make_segment(("M2",), cents=600.0, confidence=0.2),  # below threshold
        ]
        report = check_adherence(segs, sankarabharanam, min_confidence=0.5)
        # Only the high-confidence Sa is scored; it's in-raga; score = 1.0.
        assert report.adherence_score == pytest.approx(1.0)

    def test_skipped_duration_reported(self, sankarabharanam):
        segs = [
            _make_segment(("S",), cents=0.0, confidence=0.9, end=0.5),
            _make_segment(("M2",), cents=600.0, confidence=0.2, start=0.5, end=1.0),
        ]
        report = check_adherence(segs, sankarabharanam, min_confidence=0.5)
        assert report.scored_duration_seconds == pytest.approx(0.5)
        assert report.skipped_duration_seconds == pytest.approx(0.5)

    def test_invalid_min_confidence_raises(self, sankarabharanam):
        with pytest.raises(ValueError):
            check_adherence([], sankarabharanam, min_confidence=1.5)
        with pytest.raises(ValueError):
            check_adherence([], sankarabharanam, min_confidence=-0.1)


# ----- Report structure -----

class TestReport:

    def test_returns_adherence_report(self, sankarabharanam):
        segs = [_make_segment(("S",), cents=0.0)]
        report = check_adherence(segs, sankarabharanam)
        assert isinstance(report, AdherenceReport)

    def test_empty_input_returns_empty_report(self, sankarabharanam):
        report = check_adherence([], sankarabharanam)
        assert report.num_segments == 0
        assert report.num_out_of_raga == 0
        assert report.adherence_score == 0.0

    def test_per_segment_preserves_order(self, sankarabharanam):
        segs = [
            _make_segment(("S",), cents=0.0, start=0.0, end=0.3),
            _make_segment(("M2",), cents=600.0, start=0.3, end=0.6),
            _make_segment(("P",), cents=700.0, start=0.6, end=0.9),
        ]
        report = check_adherence(segs, sankarabharanam)
        assert [a.segment.start_time for a in report.per_segment] == [0.0, 0.3, 0.6]

    def test_out_of_raga_list_contains_only_failures(self, sankarabharanam):
        segs = [
            _make_segment(("S",), cents=0.0),
            _make_segment(("M2",), cents=600.0),
            _make_segment(("P",), cents=700.0),
        ]
        report = check_adherence(segs, sankarabharanam)
        assert report.num_out_of_raga == 1
        assert report.out_of_raga_segments[0].segment.swara_candidates == ("M2",)