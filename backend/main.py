"""
FastAPI backend for the Swara AI web demo.

Thin HTTP wrapper around src/pipeline.py -- no pipeline logic lives here,
so the CLI (scripts/demo.py) and the web demo call the exact same code and
can never silently diverge. This is a portfolio demo, not a production
service: validation is basic (file type, size, shruthi range) but there's
no auth, rate limiting, or persistence.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.pipeline import PipelineError, check_ragam_adherence, identify_raga, load_config
from src.swara.ragams import load_ragams

app = FastAPI(title="Swara AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_config = load_config()

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20MB
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
MIN_SHRUTHI_HZ = 80.0
MAX_SHRUTHI_HZ = 400.0


class RagaPredictionOut(BaseModel):
    raga: str
    confidence: float


class AdherenceSegmentOut(BaseModel):
    start_time: float
    end_time: float
    candidates: List[str]
    in_raga: bool
    matched_swara: Optional[str]


class AnalyzeResponse(BaseModel):
    predictions: List[RagaPredictionOut]
    adherence_score: Optional[float] = None
    adherence_ragam: Optional[str] = None
    num_segments: Optional[int] = None
    num_out_of_raga: Optional[int] = None
    segments: Optional[List[AdherenceSegmentOut]] = None


@app.get("/ragams")
def list_ragams() -> List[str]:
    """All melakarta ragam names, for the frontend's target-ragam dropdown."""
    return [r.name for r in load_ragams().all_ragams()]


@app.get("/identifier-ragas")
def list_identifier_ragas() -> List[str]:
    """
    The identifier's actual trained vocabulary (MVP-10) -- a strict subset
    of /ragams, and mostly non-overlapping (most are janya ragas not in
    the melakarta registry at all). The frontend uses this to disclose the
    identifier's real scope rather than letting a user assume it covers
    everything /ragams lists.
    """
    processed_dir = Path(_config["paths"]["data_dir"]) / "processed"
    with open(processed_dir / "mvp_ragas.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["ragas"]


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    file: UploadFile = File(...),
    shruthi_hz: float = Form(...),
    target_ragam: Optional[str] = Form(None),
) -> AnalyzeResponse:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type {ext!r}. Allowed: {sorted(ALLOWED_EXTENSIONS)}")

    if not MIN_SHRUTHI_HZ <= shruthi_hz <= MAX_SHRUTHI_HZ:
        raise HTTPException(
            status_code=400,
            detail=f"shruthi_hz must be between {MIN_SHRUTHI_HZ} and {MAX_SHRUTHI_HZ}",
        )

    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail=f"File too large (max {MAX_UPLOAD_BYTES // 1_000_000}MB)")

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = Path(tmp.name)

    try:
        try:
            predictions = identify_raga(tmp_path, shruthi_hz, _config, top_k=3)
        except PipelineError as e:
            raise HTTPException(status_code=503, detail=f"Identifier unavailable: {e}")
        except FileNotFoundError:
            raise HTTPException(status_code=400, detail="Could not decode the uploaded audio")

        response = AnalyzeResponse(
            predictions=[RagaPredictionOut(raga=p.raga, confidence=p.confidence) for p in predictions]
        )

        if target_ragam:
            try:
                report = check_ragam_adherence(tmp_path, shruthi_hz, target_ragam, _config)
            except KeyError:
                raise HTTPException(status_code=400, detail=f"Ragam {target_ragam!r} not found in registry")
            response.adherence_score = report.adherence_score
            response.adherence_ragam = report.ragam.name
            response.num_segments = report.num_segments
            response.num_out_of_raga = report.num_out_of_raga
            response.segments = [
                AdherenceSegmentOut(
                    start_time=a.segment.start_time,
                    end_time=a.segment.end_time,
                    candidates=list(a.segment.swara_candidates),
                    in_raga=a.in_raga,
                    matched_swara=a.matched_swara,
                )
                for a in report.per_segment
            ]

        return response
    finally:
        tmp_path.unlink(missing_ok=True)
