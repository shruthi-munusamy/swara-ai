"""
Combined demo: identify the raga of an audio clip, and (if a target ragam
is given, or a raga was successfully identified) check its adherence.

This is the same pipeline (src/pipeline.py) the web backend calls -- the
CLI and the web demo can never silently diverge.

Usage:
    python scripts/demo.py AUDIO SHRUTHI_HZ
    python scripts/demo.py AUDIO SHRUTHI_HZ Kalyani
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import PipelineError, check_ragam_adherence, identify_raga, load_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", type=Path, help="Path to the audio file.")
    parser.add_argument("shruthi_hz", type=float, help="Singer's tonic in Hz.")
    parser.add_argument("target_ragam", type=str, nargs="?", default=None, help="Ragam to check adherence against.")
    args = parser.parse_args()

    if not args.audio.exists():
        print(f"error: audio file not found: {args.audio}", file=sys.stderr)
        return 1

    config = load_config()

    print(f"\n=== Raga identification: {args.audio.name} ===")
    try:
        predictions = identify_raga(args.audio, args.shruthi_hz, config, top_k=3)
    except PipelineError as e:
        print(f"  identifier unavailable: {e}")
        predictions = []

    if predictions:
        for p in predictions:
            print(f"  {p.raga:20s} {p.confidence:.3f}")
    else:
        print("  No swara segments detected.")

    target = args.target_ragam or (predictions[0].raga if predictions else None)
    if target:
        print(f"\n=== Adherence check against {target} ===")
        try:
            report = check_ragam_adherence(args.audio, args.shruthi_hz, target, config)
            print(f"  Adherence score:   {report.adherence_score:.3f}")
            print(f"  Segments:          {report.num_segments}")
            print(f"  Out of raga:       {report.num_out_of_raga}")
        except KeyError:
            print(f"  error: ragam {target!r} not found in registry")
    else:
        print("\n(no target ragam given and no raga identified; skipping adherence check)")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
