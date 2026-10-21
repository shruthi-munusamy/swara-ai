"""
Identify the raga of an audio clip using the trained LSTM identifier.

Usage:
    python scripts/identify_raga.py AUDIO SHRUTHI_HZ
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import PipelineError, identify_raga, load_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", type=Path, help="Path to the audio file.")
    parser.add_argument("shruthi_hz", type=float, help="Singer's tonic in Hz.")
    parser.add_argument("--config", type=Path, default=None, help="Path to config.yaml.")
    args = parser.parse_args()

    if not args.audio.exists():
        print(f"error: audio file not found: {args.audio}", file=sys.stderr)
        return 1

    config = load_config(args.config)

    try:
        predictions = identify_raga(args.audio, args.shruthi_hz, config, top_k=3)
    except PipelineError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    if not predictions:
        print("No swara segments detected; cannot identify raga.")
        return 1

    print(f"\nTop-3 raga predictions for {args.audio.name}:")
    for p in predictions:
        print(f"  {p.raga:20s} {p.confidence:.3f}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
