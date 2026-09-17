"""
End-to-end ragam adherence check on an audio file.

Usage:
    python scripts/check_adherence.py AUDIO RAGAM SHRUTHI_HZ [options]

Examples:
    python scripts/check_adherence.py recording.wav Sankarabharanam 220
    python scripts/check_adherence.py recording.wav Kalyani 145 --model full
    python scripts/check_adherence.py recording.wav Todi 200 --json

Pipeline:
    1. Load audio at 16 kHz mono.
    2. Run CREPE to extract a pitch contour with confidence.
    3. Convert Hz to tonic-relative cents (octave-folded).
    4. Smooth and segment into stable swara regions.
    5. Look up the target ragam and check per-segment adherence.
    6. Print a human-readable report (or JSON, with --json).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Project root on sys.path so this script runs as `python scripts/...`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from src.adherence.checker import AdherenceReport, check_adherence
from src.audio.loader import load_audio
from src.pitch.cents import to_cents_contour
from src.pitch.extractor import extract_pitch
from src.swara.ragams import load_ragams
from src.swara.segmenter import segment_swaras


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def _load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _format_human_report(report: AdherenceReport, audio_path: Path) -> str:
    """Render an AdherenceReport as plain text for terminal output."""
    lines = []
    lines.append("")
    lines.append(f"Audio:   {audio_path.name}")
    lines.append(f"Ragam:   {report.ragam.name} (mela {report.ragam.number})")
    lines.append(f"Swaras:  {' '.join(report.ragam.swaras)}")
    lines.append("")
    lines.append(f"Adherence score:     {report.adherence_score:.3f}")
    lines.append(f"Segments analyzed:   {report.num_segments}")
    lines.append(f"Out of raga:         {report.num_out_of_raga}")
    lines.append(f"Scored duration:     {report.scored_duration_seconds:.2f}s")
    lines.append(f"Skipped (low conf):  {report.skipped_duration_seconds:.2f}s")

    if report.out_of_raga_segments:
        lines.append("")
        lines.append("Out-of-raga regions:")
        for ann in report.out_of_raga_segments:
            seg = ann.segment
            candidates = ", ".join(seg.swara_candidates)
            lines.append(
                f"  {seg.start_time:6.2f}s - {seg.end_time:6.2f}s  "
                f"({seg.duration_seconds:.2f}s)  "
                f"candidates: {candidates:<10s}  "
                f"cents: {seg.cents:6.1f}  "
                f"conf: {seg.confidence:.2f}"
            )

    lines.append("")
    return "\n".join(lines)


def _format_json_report(report: AdherenceReport, audio_path: Path) -> str:
    """Render an AdherenceReport as a JSON string. Useful for downstream tools."""
    payload = {
        "audio": str(audio_path),
        "ragam": {
            "number": report.ragam.number,
            "name": report.ragam.name,
            "swaras": report.ragam.swaras,
        },
        "adherence_score": report.adherence_score,
        "num_segments": report.num_segments,
        "num_out_of_raga": report.num_out_of_raga,
        "scored_duration_seconds": report.scored_duration_seconds,
        "skipped_duration_seconds": report.skipped_duration_seconds,
        "segments": [
            {
                "start_time": ann.segment.start_time,
                "end_time": ann.segment.end_time,
                "candidates": list(ann.segment.swara_candidates),
                "cents": ann.segment.cents,
                "confidence": ann.segment.confidence,
                "in_raga": ann.in_raga,
                "matched_swara": ann.matched_swara,
                "deviation_cents": ann.deviation_cents,
            }
            for ann in report.per_segment
        ],
    }
    return json.dumps(payload, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", type=Path, help="Path to the audio file.")
    parser.add_argument("ragam", type=str, help="Target ragam name (e.g. Sankarabharanam, Kalyani).")
    parser.add_argument("shruthi_hz", type=float, help="Singer's tonic in Hz (e.g. 220 for typical female pitch).")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to config.yaml.")
    parser.add_argument(
        "--model",
        choices=["tiny", "small", "medium", "large", "full"],
        default=None,
        help="CREPE model capacity. Overrides config.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of human-readable text.")
    args = parser.parse_args()

    if not args.audio.exists():
        print(f"error: audio file not found: {args.audio}", file=sys.stderr)
        return 1

    config = _load_config(args.config)
    audio_cfg = config["audio"]
    pitch_cfg = config["pitch"]
    swara_cfg = config["swara"]
    adherence_cfg = config["adherence"]

    crepe_model = args.model if args.model else pitch_cfg["crepe_model"]

    # 1. Audio
    clip = load_audio(args.audio, sample_rate=audio_cfg["sample_rate"])
    if not args.json:
        print(f"Loaded {args.audio.name}: {clip.duration_seconds:.2f}s @ {clip.sample_rate}Hz", file=sys.stderr)

    # 2. Pitch
    if not args.json:
        print(f"Extracting pitch (CREPE {crepe_model})...", file=sys.stderr)
    contour = extract_pitch(
        clip,
        model=crepe_model,
        step_size_ms=pitch_cfg["step_size_ms"],
    )

    # 3. Cents
    cents_contour = to_cents_contour(
        contour,
        shruthi_hz=args.shruthi_hz,
        confidence_threshold=pitch_cfg["confidence_threshold"],
        octave_fold=True,
    )

    # 4. Segments
    segments = segment_swaras(
        cents_contour,
        tolerance_cents=swara_cfg["tolerance_cents"],
        min_stable_ms=swara_cfg["min_stable_ms"],
        smoothing_window=swara_cfg["smoothing_window"],
    )
    if not args.json:
        print(f"Extracted {len(segments)} swara segments.", file=sys.stderr)

    # 5. Adherence
    try:
        ragam = load_ragams().get_by_name(args.ragam)
    except KeyError:
        print(f"error: ragam {args.ragam!r} not found. Try 'Sankarabharanam', 'Kalyani', etc.", file=sys.stderr)
        return 1

    report = check_adherence(
        segments,
        ragam,
        min_confidence=adherence_cfg["min_confidence"],
    )

    # 6. Output
    if args.json:
        print(_format_json_report(report, args.audio))
    else:
        print(_format_human_report(report, args.audio))

    return 0


if __name__ == "__main__":
    sys.exit(main())