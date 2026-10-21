"""
Profile end-to-end pipeline latency, stage by stage, on a reference clip.

Usage:
    python scripts/profile_latency.py
    python scripts/profile_latency.py --clip data/samples/demo_clip.wav --runs 10

Outputs:
    models/latency_profile.md
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.audio.loader import load_audio
from src.identifier.dataset import segment_to_class_index
from src.pipeline import load_identifier_model, load_config
from src.pitch.cents import to_cents_contour
from src.pitch.extractor import extract_pitch
from src.swara.segmenter import segment_swaras

DEFAULT_CLIP = Path("data/samples/demo_clip.wav")


def _percentile(values, p):
    return float(np.percentile(values, p))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clip", type=Path, default=DEFAULT_CLIP)
    parser.add_argument("--shruthi-hz", type=float, default=220.0)
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()

    config = load_config()
    audio_cfg = config["audio"]
    pitch_cfg = config["pitch"]
    swara_cfg = config["swara"]

    stage_times = {"audio_load": [], "crepe": [], "cents": [], "segment": [], "lstm": [], "total": []}

    model, label_names = load_identifier_model(config)

    for _ in range(args.runs):
        t0 = time.perf_counter()

        t_start = time.perf_counter()
        clip = load_audio(args.clip, sample_rate=audio_cfg["sample_rate"])
        stage_times["audio_load"].append(time.perf_counter() - t_start)

        t_start = time.perf_counter()
        contour = extract_pitch(clip, model=pitch_cfg["crepe_model"], step_size_ms=pitch_cfg["step_size_ms"])
        stage_times["crepe"].append(time.perf_counter() - t_start)

        t_start = time.perf_counter()
        cents_contour = to_cents_contour(
            contour, shruthi_hz=args.shruthi_hz,
            confidence_threshold=pitch_cfg["confidence_threshold"], octave_fold=True,
        )
        stage_times["cents"].append(time.perf_counter() - t_start)

        t_start = time.perf_counter()
        segments = segment_swaras(
            cents_contour, tolerance_cents=swara_cfg["tolerance_cents"],
            min_stable_ms=swara_cfg["min_stable_ms"], smoothing_window=swara_cfg["smoothing_window"],
        )
        stage_times["segment"].append(time.perf_counter() - t_start)

        t_start = time.perf_counter()
        if segments:
            indices = [segment_to_class_index(seg.swara_candidates) for seg in segments]
            seq = torch.tensor(indices, dtype=torch.long).unsqueeze(0)
            length = torch.tensor([len(indices)])
            with torch.no_grad():
                model(seq, length)
        stage_times["lstm"].append(time.perf_counter() - t_start)

        stage_times["total"].append(time.perf_counter() - t0)

    clip_duration = load_audio(args.clip, sample_rate=audio_cfg["sample_rate"]).duration_seconds

    lines = [
        f"Latency profile -- {args.runs} runs on a {clip_duration:.1f}s clip "
        f"(CREPE model: {pitch_cfg['crepe_model']})",
        "",
        "| Stage       | p50 (ms) | p95 (ms) |",
        "|-------------|----------|----------|",
    ]
    for stage in ["audio_load", "crepe", "cents", "segment", "lstm", "total"]:
        vals_ms = [v * 1000 for v in stage_times[stage]]
        lines.append(f"| {stage:<11s} | {_percentile(vals_ms, 50):8.2f} | {_percentile(vals_ms, 95):8.2f} |")

    report = "\n".join(lines)
    print("\n" + report)

    models_dir = Path(config["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    out_path = models_dir / "latency_profile.md"
    out_path.write_text(report + "\n", encoding="utf-8")
    print(f"\nWrote {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
