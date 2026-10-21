"""Integration tests for the FastAPI backend."""

from pathlib import Path

from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)

SAMPLE_CLIP = Path(__file__).resolve().parents[1] / "data" / "samples" / "demo_clip.wav"


class TestAnalyzeEndpoint:

    def test_returns_predictions_for_valid_clip(self):
        with open(SAMPLE_CLIP, "rb") as f:
            response = client.post(
                "/analyze",
                files={"file": ("demo_clip.wav", f, "audio/wav")},
                data={"shruthi_hz": "220"},
            )
        assert response.status_code == 200
        body = response.json()
        assert "predictions" in body
        assert len(body["predictions"]) > 0

    def test_with_target_ragam_returns_adherence_report(self):
        with open(SAMPLE_CLIP, "rb") as f:
            response = client.post(
                "/analyze",
                files={"file": ("demo_clip.wav", f, "audio/wav")},
                data={"shruthi_hz": "220", "target_ragam": "Sankarabharanam"},
            )
        assert response.status_code == 200
        body = response.json()
        assert body["adherence_ragam"] == "Sankarabharanam"
        assert 0.0 <= body["adherence_score"] <= 1.0
        assert body["num_segments"] > 0

    def test_without_target_ragam_omits_adherence_fields(self):
        with open(SAMPLE_CLIP, "rb") as f:
            response = client.post(
                "/analyze",
                files={"file": ("demo_clip.wav", f, "audio/wav")},
                data={"shruthi_hz": "220"},
            )
        body = response.json()
        assert body["adherence_score"] is None

    def test_rejects_unsupported_file_type(self):
        response = client.post(
            "/analyze",
            files={"file": ("test.txt", b"not audio", "text/plain")},
            data={"shruthi_hz": "220"},
        )
        assert response.status_code == 400

    def test_rejects_out_of_range_shruthi(self):
        with open(SAMPLE_CLIP, "rb") as f:
            response = client.post(
                "/analyze",
                files={"file": ("demo_clip.wav", f, "audio/wav")},
                data={"shruthi_hz": "9999"},
            )
        assert response.status_code == 400

    def test_unknown_target_ragam_returns_400(self):
        with open(SAMPLE_CLIP, "rb") as f:
            response = client.post(
                "/analyze",
                files={"file": ("demo_clip.wav", f, "audio/wav")},
                data={"shruthi_hz": "220", "target_ragam": "NotARealRagam"},
            )
        assert response.status_code == 400


class TestRagamsEndpoint:

    def test_returns_known_ragam_names(self):
        response = client.get("/ragams")
        assert response.status_code == 200
        names = response.json()
        assert "Kalyani" in names
        assert len(names) == 72
