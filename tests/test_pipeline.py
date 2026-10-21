"""Tests for the shared end-to-end pipeline used by the CLI and backend."""

from pathlib import Path

import pytest

from src.pipeline import (
    PipelineError,
    RagaPrediction,
    check_ragam_adherence,
    extract_segments,
    identify_raga,
    load_config,
)

SAMPLE_CLIP = Path(__file__).resolve().parents[1] / "data" / "samples" / "demo_clip.wav"


@pytest.fixture(scope="module")
def config():
    return load_config()


class TestLoadConfig:

    def test_loads_default_config(self, config):
        assert "audio" in config
        assert "identifier" in config


class TestExtractSegments:

    def test_returns_segments_for_real_clip(self, config):
        segments = extract_segments(SAMPLE_CLIP, shruthi_hz=220.0, config=config)
        assert len(segments) > 0

    def test_missing_file_raises(self, config):
        with pytest.raises(FileNotFoundError):
            extract_segments(Path("does_not_exist.wav"), 220.0, config)


class TestCheckRagamAdherence:

    def test_returns_report_for_known_ragam(self, config):
        report = check_ragam_adherence(SAMPLE_CLIP, 220.0, "Sankarabharanam", config)
        assert report.ragam.name == "Sankarabharanam"
        assert 0.0 <= report.adherence_score <= 1.0

    def test_unknown_ragam_raises_keyerror(self, config):
        with pytest.raises(KeyError):
            check_ragam_adherence(SAMPLE_CLIP, 220.0, "NotARealRagam", config)


class TestIdentifyRaga:

    def test_returns_predictions_when_model_available(self, config):
        try:
            predictions = identify_raga(SAMPLE_CLIP, 220.0, config, top_k=3)
        except PipelineError:
            pytest.skip("identifier checkpoint/mvp_ragas.yaml not present in this environment")
            return
        assert len(predictions) <= 3
        for p in predictions:
            assert isinstance(p, RagaPrediction)
            assert 0.0 <= p.confidence <= 1.0

    def test_predictions_sorted_by_confidence_descending(self, config):
        try:
            predictions = identify_raga(SAMPLE_CLIP, 220.0, config, top_k=3)
        except PipelineError:
            pytest.skip("identifier checkpoint/mvp_ragas.yaml not present in this environment")
            return
        confidences = [p.confidence for p in predictions]
        assert confidences == sorted(confidences, reverse=True)
