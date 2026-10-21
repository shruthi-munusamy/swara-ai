"""
Shared end-to-end pipeline: audio -> raga identification and/or ragam
adherence check.

Both CLI scripts (identify_raga.py, demo.py) and the FastAPI backend
(backend/main.py) call into this module rather than reimplementing the
pipeline -- one code path, so the web demo and CLI can never silently
diverge from each other.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import yaml

from src.adherence.checker import AdherenceReport, check_adherence
from src.audio.loader import load_audio
from src.identifier.dataset import NUM_SWARA_CLASSES, segment_to_class_index
from src.identifier.model import RagaLSTM
from src.pitch.cents import to_cents_contour
from src.pitch.extractor import extract_pitch
from src.swara.ragams import load_ragams
from src.swara.segmenter import SwaraSegment, segment_swaras

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

# Module-level cache: the identifier checkpoint and label list only need to
# be loaded once per process, not once per request (the backend serves many
# requests from one process; reloading a ~2.5MB checkpoint and re-reading
# YAML on every call would be pure waste).
_MODEL_CACHE: dict = {}


def load_config(path: Optional[Path] = None) -> dict:
    path = path or DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass(frozen=True)
class RagaPrediction:
    raga: str
    confidence: float


class PipelineError(RuntimeError):
    """Raised when a pipeline stage fails in a way callers should surface distinctly."""
    pass


def extract_segments(audio_path: Path, shruthi_hz: float, config: dict) -> List[SwaraSegment]:
    """Run the shared audio -> pitch -> cents -> segments pipeline."""
    audio_cfg = config["audio"]
    pitch_cfg = config["pitch"]
    swara_cfg = config["swara"]

    clip = load_audio(audio_path, sample_rate=audio_cfg["sample_rate"])
    contour = extract_pitch(clip, model=pitch_cfg["crepe_model"], step_size_ms=pitch_cfg["step_size_ms"])
    cents_contour = to_cents_contour(
        contour,
        shruthi_hz=shruthi_hz,
        confidence_threshold=pitch_cfg["confidence_threshold"],
        octave_fold=True,
    )
    return segment_swaras(
        cents_contour,
        tolerance_cents=swara_cfg["tolerance_cents"],
        min_stable_ms=swara_cfg["min_stable_ms"],
        smoothing_window=swara_cfg["smoothing_window"],
    )


def check_ragam_adherence(audio_path: Path, shruthi_hz: float, ragam_name: str, config: dict) -> AdherenceReport:
    """Run the shared pipeline, then score the result against a target ragam."""
    segments = extract_segments(audio_path, shruthi_hz, config)
    ragam = load_ragams().get_by_name(ragam_name)
    return check_adherence(segments, ragam, min_confidence=config["adherence"]["min_confidence"])


def load_identifier_model(config: dict) -> Tuple[RagaLSTM, List[str]]:
    cache_key = str(Path(config["paths"]["models_dir"]).resolve())
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    processed_dir = Path(config["paths"]["data_dir"]) / "processed"
    mvp_path = processed_dir / "mvp_ragas.yaml"
    if not mvp_path.exists():
        raise PipelineError(f"{mvp_path} not found; run scripts/prepare_identifier_data.py first.")
    with open(mvp_path, "r", encoding="utf-8") as f:
        label_names = yaml.safe_load(f)["ragas"]

    checkpoint_path = Path(config["paths"]["models_dir"]) / "identifier_best.pt"
    if not checkpoint_path.exists():
        raise PipelineError(f"{checkpoint_path} not found; run scripts/train_identifier.py first.")

    id_cfg = config["identifier"]
    model = RagaLSTM(
        num_swara_classes=NUM_SWARA_CLASSES,
        num_ragas=len(label_names),
        embedding_dim=id_cfg["embedding_dim"],
        hidden_size=id_cfg["hidden_size"],
        num_layers=id_cfg["num_layers"],
        dropout=id_cfg["dropout"],
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    _MODEL_CACHE[cache_key] = (model, label_names)
    return model, label_names


def identify_raga(audio_path: Path, shruthi_hz: float, config: dict, top_k: int = 3) -> List[RagaPrediction]:
    """
    Run the shared pipeline, then classify the resulting swara sequence
    with the trained LSTM identifier. Returns the top-k predictions,
    highest confidence first. Empty list if no swara segments were
    extracted (e.g. silent or unvoiced clip).
    """
    segments = extract_segments(audio_path, shruthi_hz, config)
    if not segments:
        return []

    indices = [segment_to_class_index(seg.swara_candidates) for seg in segments]
    model, label_names = load_identifier_model(config)

    seq = torch.tensor(indices, dtype=torch.long).unsqueeze(0)
    length = torch.tensor([len(indices)])
    with torch.no_grad():
        logits = model(seq, length)
        probs = torch.softmax(logits, dim=1).numpy()[0]

    order = probs.argsort()[::-1][:top_k]
    return [RagaPrediction(raga=label_names[i], confidence=float(probs[i])) for i in order]
