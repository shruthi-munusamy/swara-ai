"""
Download the Saraga Carnatic dataset (CompMusic / MTG-UPF) from Zenodo.

Saraga 1.5 Carnatic is a single ~14GB zip containing full concert audio
(multitrack: vocal mix + separated mridangam/violin/ghatam where available),
tonic annotations (ctonic.txt), and per-track metadata JSON (raaga, artist,
form, etc). We only need the vocal mix audio, the tonic, and the raaga label
per track; the rest of the zip is not used but there's no way to fetch a
subset from Zenodo's single-file record, so we download the whole thing once.

Usage:
    python scripts/download_saraga.py
    python scripts/download_saraga.py --dest data/raw/saraga

Resumable: if a partial zip exists (from a prior interrupted run), resumes
via an HTTP Range request rather than restarting.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

import requests
from tqdm import tqdm

ZENODO_RECORD_URL = (
    "https://zenodo.org/api/records/4301737/files/saraga1.5_carnatic.zip/content"
)
EXPECTED_ZIP_SIZE_BYTES = 14_376_627_433
ZIP_FILENAME = "saraga1.5_carnatic.zip"

DEFAULT_DEST = Path("data/raw/saraga")

CHUNK_SIZE = 1024 * 1024  # 1 MB


class DownloadError(RuntimeError):
    """Raised when the Saraga download or extraction fails."""
    pass


MAX_RETRIES = 20
RETRY_BACKOFF_SECONDS = 5
REQUEST_TIMEOUT_SECONDS = 30


def _download_with_resume(url: str, dest_path: Path, expected_size: int) -> None:
    """
    Download `url` to `dest_path`, resuming from an existing partial file
    via HTTP Range if one is present and smaller than the expected size.

    Zenodo's connection has proven flaky in practice (read timeouts and
    dropped connections mid-transfer, observed well before the full 14GB
    transferred). Rather than relying on the operator to notice a dead
    process and re-invoke the script, retries happen inside this function:
    a dropped connection re-opens a fresh Range request from the current
    file offset, up to `MAX_RETRIES` times, with a fixed backoff. This is
    the same resume mechanism used for a fully-restarted process, just
    triggered automatically instead of requiring a re-run.
    """
    if dest_path.exists() and dest_path.stat().st_size > expected_size:
        raise DownloadError(
            f"{dest_path} is {dest_path.stat().st_size:,} bytes, larger than "
            f"the expected {expected_size:,}. It may be corrupt; delete it and re-run."
        )

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    bar = tqdm(total=expected_size, unit="B", unit_scale=True, desc=ZIP_FILENAME)
    try:
        attempt = 0
        while True:
            existing_size = dest_path.stat().st_size if dest_path.exists() else 0
            bar.n = existing_size
            bar.refresh()

            if existing_size >= expected_size:
                break

            headers = {"Range": f"bytes={existing_size}-"} if existing_size > 0 else {}
            mode = "ab" if existing_size > 0 else "wb"

            try:
                with requests.get(
                    url, headers=headers, stream=True, timeout=REQUEST_TIMEOUT_SECONDS
                ) as resp:
                    resp.raise_for_status()
                    with open(dest_path, mode) as f:
                        for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                            if chunk:
                                f.write(chunk)
                                bar.update(len(chunk))
                attempt = 0  # reset backoff after any successful chunk of progress
            except requests.RequestException as e:
                attempt += 1
                if attempt > MAX_RETRIES:
                    raise DownloadError(
                        f"Gave up after {MAX_RETRIES} retries. Last error: {e}. "
                        f"Re-run the script later to resume from "
                        f"{dest_path.stat().st_size if dest_path.exists() else 0:,} bytes."
                    ) from e
                wait = RETRY_BACKOFF_SECONDS * attempt
                print(
                    f"\nConnection dropped ({e}); retrying in {wait}s "
                    f"(attempt {attempt}/{MAX_RETRIES})...",
                    file=sys.stderr,
                )
                time.sleep(wait)
    finally:
        bar.close()

    final_size = dest_path.stat().st_size
    if final_size != expected_size:
        raise DownloadError(
            f"Downloaded {final_size:,} bytes, expected {expected_size:,}. "
            f"Re-run to resume; if this persists the Zenodo record may have changed."
        )


def _extract(zip_path: Path, dest_dir: Path) -> None:
    """Extract the zip if the destination doesn't already look extracted."""
    marker = dest_dir / ".extracted"
    if marker.exists():
        print(f"Already extracted (found {marker}). Skipping.")
        return

    print(f"Extracting {zip_path.name} to {dest_dir}...")
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.namelist()
        for member in tqdm(members, desc="extracting", unit="file"):
            zf.extract(member, dest_dir)

    marker.write_text("")


def _find_metadata_files(root: Path) -> list[Path]:
    # __MACOSX/ holds AppleDouble resource-fork sidecar files (._*.json) --
    # binary garbage created by macOS zip tooling, not real track metadata.
    return sorted(p for p in root.rglob("*.json") if "__MACOSX" not in p.parts)


def _extract_raaga_name(metadata: dict) -> str | None:
    """
    Pull the raaga name out of a Saraga metadata JSON dict.

    `raaga` is documented as a list of dicts (a track can nominally list
    more than one, though in practice concerts are single-raga). We take
    the first entry's `name` field. Returns None if absent or malformed —
    tracks without a parseable raaga are excluded from the count rather
    than crashing the whole summary.
    """
    raaga = metadata.get("raaga")
    if not raaga or not isinstance(raaga, list):
        return None
    first = raaga[0]
    if isinstance(first, dict):
        return first.get("name")
    if isinstance(first, str):
        return first
    return None


def _print_summary(dest_dir: Path) -> None:
    metadata_files = _find_metadata_files(dest_dir)
    audio_files = [
        p for p in list(dest_dir.rglob("*.mp3")) + list(dest_dir.rglob("*.wav"))
        if "__MACOSX" not in p.parts
    ]

    total_bytes = sum(f.stat().st_size for f in dest_dir.rglob("*") if f.is_file())

    print()
    print(f"Extracted size:  {total_bytes / 1e9:.2f} GB")
    print(f"Metadata files:  {len(metadata_files)}")
    print(f"Audio files:     {len(audio_files)}")

    raaga_counts: Counter[str] = Counter()
    unparsed = 0
    for mf in metadata_files:
        try:
            with open(mf, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            unparsed += 1
            continue
        name = _extract_raaga_name(metadata)
        if name is None:
            unparsed += 1
            continue
        raaga_counts[name] += 1

    print(f"Unparsed/no-raaga metadata files: {unparsed}")
    print()
    print("Per-raga track counts (top 20):")
    for name, count in raaga_counts.most_common(20):
        print(f"  {count:4d}  {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help=f"Destination directory (default: {DEFAULT_DEST})",
    )
    args = parser.parse_args()

    dest_dir: Path = args.dest
    zip_path = dest_dir / ZIP_FILENAME

    try:
        _download_with_resume(ZENODO_RECORD_URL, zip_path, EXPECTED_ZIP_SIZE_BYTES)
        _extract(zip_path, dest_dir)
    except (requests.RequestException, DownloadError, zipfile.BadZipFile) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    _print_summary(dest_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
