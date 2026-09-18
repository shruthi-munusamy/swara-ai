"""
Prepare the MVP raga set and train/val/test splits for the identifier.

This is a one-time (re-runnable, deterministic) step between raw Saraga
data and training: it decides *which* 10 ragas the MVP identifier will
classify (a data-driven choice -- the 10 most-represented in the corpus,
not a theory-driven pick) and writes reproducible splits.

Usage:
    python scripts/prepare_identifier_data.py
    python scripts/prepare_identifier_data.py --saraga-root data/raw/saraga --num-ragas 10

Outputs:
    data/processed/mvp_ragas.yaml  -- chosen ragas + their track counts
    data/processed/splits.yaml     -- track_id lists for train/val/test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.identifier.dataset import build_mvp_raga_list, discover_tracks, raga_counts, stratified_split

DEFAULT_SARAGA_ROOT = Path("data/raw/saraga")
DEFAULT_OUTPUT_DIR = Path("data/processed")
DEFAULT_NUM_RAGAS = 10
DEFAULT_SEED = 42


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--saraga-root", type=Path, default=DEFAULT_SARAGA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-ragas", type=int, default=DEFAULT_NUM_RAGAS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    if not args.saraga_root.exists():
        print(f"error: {args.saraga_root} not found. Run scripts/download_saraga.py first.", file=sys.stderr)
        return 1

    print(f"Scanning {args.saraga_root} for usable tracks...")
    tracks = discover_tracks(args.saraga_root)
    print(f"Found {len(tracks)} usable tracks.")

    counts = raga_counts(tracks)
    print(f"\n{len(counts)} distinct ragas. Per-raga counts (top 30):")
    for name, count in counts.most_common(30):
        print(f"  {count:4d}  {name}")

    mvp_ragas = build_mvp_raga_list(tracks, top_k=args.num_ragas)
    mvp_set = set(mvp_ragas)
    print(f"\nSelected top {args.num_ragas} ragas by track count: {mvp_ragas}")

    filtered = [t for t in tracks if t.raga in mvp_set]
    print(f"{len(filtered)} tracks remain after filtering to the MVP raga set.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    mvp_ragas_path = args.output_dir / "mvp_ragas.yaml"
    with open(mvp_ragas_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "ragas": mvp_ragas,
                "counts": {name: counts[name] for name in mvp_ragas},
                "seed": args.seed,
            },
            f,
            sort_keys=False,
        )
    print(f"Wrote {mvp_ragas_path}")

    train, val, test = stratified_split(filtered, seed=args.seed)
    splits_path = args.output_dir / "splits.yaml"
    with open(splits_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "seed": args.seed,
                "train": [t.track_id for t in train],
                "val": [t.track_id for t in val],
                "test": [t.track_id for t in test],
            },
            f,
            sort_keys=False,
        )
    print(f"Wrote {splits_path} (train={len(train)}, val={len(val)}, test={len(test)})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
