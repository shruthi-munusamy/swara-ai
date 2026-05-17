"""
Visualize CREPE pitch extraction on a real audio file.

Usage:
    python scripts/sanity_check_pitch.py path/to/recording.wav

Prints summary statistics, then saves a PNG plot of pitch and confidence
versus time. Use this to sanity-check the extractor on any new audio before
running it through downstream pipeline stages.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Save to file, don't open a window.
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path so this script works when run as `python scripts/...`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.audio.loader import load_audio
from src.pitch.extractor import extract_pitch


def main(audio_path: Path, model: str, threshold: float, output_path: Path) -> None:
    clip = load_audio(audio_path)
    print(f"Loaded {audio_path.name}: {clip.duration_seconds:.2f}s @ {clip.sample_rate}Hz")

    contour = extract_pitch(clip, model=model)
    voiced = contour.voiced_mask(threshold)

    print(f"Frames: {contour.num_frames}")
    print(f"Voiced frames (conf >= {threshold}): {int(voiced.sum())} ({100 * voiced.mean():.1f}%)")
    if voiced.any():
        f_voiced = contour.frequencies_hz[voiced]
        print(f"Voiced pitch range: {f_voiced.min():.1f} - {f_voiced.max():.1f} Hz")
        print(f"Voiced pitch median: {np.median(f_voiced):.1f} Hz")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

    # Pitch plot, colored by confidence.
    sc = ax1.scatter(
        contour.times,
        contour.frequencies_hz,
        c=contour.confidence,
        cmap="viridis",
        s=4,
        vmin=0.0,
        vmax=1.0,
    )
    ax1.set_ylabel("Frequency (Hz)")
    ax1.set_yscale("log")
    ax1.set_title(f"CREPE pitch contour — {audio_path.name} (model={model})")
    plt.colorbar(sc, ax=ax1, label="confidence")

    # Confidence plot.
    ax2.plot(contour.times, contour.confidence, linewidth=0.8)
    ax2.axhline(threshold, color="red", linestyle="--", linewidth=0.8, label=f"threshold={threshold}")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Confidence")
    ax2.set_ylim(0.0, 1.05)
    ax2.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="Path to audio file.")
    parser.add_argument(
        "--model",
        choices=["tiny", "small", "medium", "large", "full"],
        default="tiny",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to save the PNG. Defaults to <audio_stem>_pitch.png.",
    )
    args = parser.parse_args()

    output = args.output if args.output else args.audio.with_name(
        f"{args.audio.stem}_pitch.png"
    )
    main(args.audio, args.model, args.threshold, output)