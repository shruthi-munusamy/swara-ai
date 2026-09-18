"""
Saraga-backed PyTorch Dataset for raga identification.

Each example is a track: raw audio -> shared pipeline (CREPE pitch ->
tonic-relative cents -> swara segmentation) -> a sequence of swara-class
indices, paired with the track's raga label.

Feature representation (locked decision, see design log):
    Segmented swara sequence, not raw frame-rate cents. This is the same
    representation the adherence checker consumes -- one pipeline, two
    heads -- and it's what a Carnatic listener actually attends to (which
    swara was sung, not the raw pitch curve). The identifier sees swara
    *positions*, not swara *names*: shared-cents-position swaras (R2/G1,
    R3/G2, D2/N1, D3/N2) collapse to one class each, because resolving them
    to a specific name requires already knowing the raga (that's what the
    adherence checker does using the *target* raga) -- doing that here would
    leak the label into the input.

Confidence handling: low-confidence CREPE frames are excluded at the
segmentation stage (segment_swaras only forms segments from voiced frames;
voicing is thresholded upstream in to_cents_contour). The identifier sees
only segments that already cleared that bar. This is the same policy the
adherence checker uses, applied for the same reason: cleaner signal, one
confidence-handling policy instead of two.

Variable-length sequences: tracks differ wildly in duration and swara count.
Handled by padding + explicit lengths in `collate_fn`, consumed by
`pack_padded_sequence` in the model (C12) -- not truncation, which would
throw away signal from longer performances.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from src.audio.loader import load_audio
from src.pitch.cents import to_cents_contour
from src.pitch.extractor import extract_pitch
from src.swara.definitions import SWARASTHANA_CENTS
from src.swara.segmenter import segment_swaras

# The 12 unique cents positions swarasthanas occupy, in ascending order.
# All SWARASTHANA_CENTS values are multiples of 100 by construction (12-TET),
# so index = cents / 100 is a valid, collision-free mapping -- no lookup
# table needed. Kept as an explicit sorted list (rather than relying on the
# arithmetic silently) so the vocabulary size and ordering are visible here.
SWARA_CLASSES: List[float] = sorted(set(SWARASTHANA_CENTS.values()))
NUM_SWARA_CLASSES: int = len(SWARA_CLASSES)
assert NUM_SWARA_CLASSES == 12, "expected 12 unique swarasthana positions"


class DatasetError(ValueError):
    """Raised when dataset construction or feature extraction fails."""
    pass


def cents_to_class_index(cents: float) -> int:
    """
    Map an exact swarasthana cents position (0, 100, ..., 1100) to its
    0-11 class index. Raises if `cents` isn't one of the 12 canonical
    positions -- callers must pass a value already snapped to a swarasthana
    (e.g. via `swara_to_cents` on one of a segment's `swara_candidates`),
    not a raw measured cents value.
    """
    index = round(cents / 100.0)
    if index < 0 or index > 11 or abs(cents - index * 100.0) > 1e-6:
        raise DatasetError(
            f"cents={cents} is not a canonical swarasthana position "
            f"(expected a multiple of 100 in [0, 1100])."
        )
    return index


def segment_to_class_index(swara_candidates: Sequence[str]) -> int:
    """
    Map a SwaraSegment's candidate tuple to its 12-way class index.

    All candidates in the tuple share the same cents position by
    construction (that's why the segmenter grouped them), so any one of
    them resolves to the same index.
    """
    from src.swara.definitions import swara_to_cents  # local import: avoid cycle at module load

    if not swara_candidates:
        raise DatasetError("swara_candidates is empty; cannot resolve a class index")
    return cents_to_class_index(swara_to_cents(swara_candidates[0]))


@dataclass(frozen=True)
class TrackInfo:
    """
    A single Saraga track's location and ground-truth labels, discovered
    from the extracted dataset directory.

    Fields:
        track_id: unique identifier (relative directory path, used as the
            cache filename stem).
        audio_path: path to the vocal mix audio file.
        tonic_hz: the track's annotated tonic (shruthi), from ctonic.txt.
        raga: raga name as it appears in Saraga metadata.
    """
    track_id: str
    audio_path: Path
    tonic_hz: float
    raga: str


def _read_tonic(ctonic_path: Path) -> float:
    """Saraga ctonic files hold a single float (Hz) as their only content."""
    text = ctonic_path.read_text(encoding="utf-8").strip()
    try:
        return float(text.split()[0])
    except (ValueError, IndexError) as e:
        raise DatasetError(f"Could not parse tonic from {ctonic_path}: {text!r}") from e


def _extract_raaga_name(metadata: dict) -> Optional[str]:
    """Same parsing rule as scripts/download_saraga.py's summary printer."""
    raaga = metadata.get("raaga")
    if not raaga or not isinstance(raaga, list):
        return None
    first = raaga[0]
    if isinstance(first, dict):
        return first.get("name")
    if isinstance(first, str):
        return first
    return None


def discover_tracks(saraga_root: Path) -> List[TrackInfo]:
    """
    Scan an extracted Saraga directory for tracks with a complete set of
    (audio, tonic, raga label).

    A track directory is expected to contain:
        - exactly one metadata JSON with a parseable `raaga`
        - exactly one ctonic file (name contains "ctonic")
        - an audio file to run pitch extraction on. When an isolated vocal
          stem is available (Saraga ships multitrack separations for a
          subset of tracks), it's preferred over the full mix: CREPE is a
          monophonic pitch tracker, and a mix containing violin + mridangam
          + ghatam produces substantially noisier pitch estimates than a
          solo vocal channel. Falls back to the mix (a plain "<title>.mp3"
          with no instrument suffix) when no vocal stem exists.

    Track directories missing any of these, or with an ambiguous audio
    file, are skipped with a note printed to stdout rather than raising --
    Saraga has some incomplete/partial tracks and the dataset should degrade
    gracefully rather than failing the whole scan over one bad entry.

    Returns:
        List of TrackInfo, one per usable track, sorted by track_id.
    """
    tracks: List[TrackInfo] = []
    skipped = 0

    for metadata_path in sorted(saraga_root.rglob("*.json")):
        track_dir = metadata_path.parent
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except (json.JSONDecodeError, OSError):
            skipped += 1
            continue

        raga = _extract_raaga_name(metadata)
        if raga is None:
            skipped += 1
            continue

        ctonic_candidates = sorted(track_dir.glob("*ctonic*"))
        if len(ctonic_candidates) != 1:
            skipped += 1
            continue

        audio_files = list(track_dir.glob("*.mp3")) + list(track_dir.glob("*.wav"))
        vocal_candidates = sorted(p for p in audio_files if "vocal" in p.stem.lower())
        mix_candidates = sorted(
            p for p in audio_files
            if "multitrack" not in p.stem.lower() and "vocal" not in p.stem.lower()
        )
        if len(vocal_candidates) == 1:
            chosen_audio = vocal_candidates[0]
        elif len(vocal_candidates) == 0 and len(mix_candidates) == 1:
            chosen_audio = mix_candidates[0]
        else:
            # Ambiguous (multiple vocal stems, or no vocal stem and multiple/no
            # mix candidates) -- skip rather than guess wrong.
            skipped += 1
            continue

        try:
            tonic_hz = _read_tonic(ctonic_candidates[0])
        except DatasetError:
            skipped += 1
            continue

        track_id = str(track_dir.relative_to(saraga_root))
        tracks.append(
            TrackInfo(
                track_id=track_id,
                audio_path=chosen_audio,
                tonic_hz=tonic_hz,
                raga=raga,
            )
        )

    if skipped:
        print(f"discover_tracks: skipped {skipped} incomplete/ambiguous track directories")

    return tracks


def raga_counts(tracks: Sequence[TrackInfo]) -> Counter:
    """Per-raga track counts, most-common first when iterated via .most_common()."""
    return Counter(t.raga for t in tracks)


def compute_swara_sequence(
    track: TrackInfo,
    sample_rate: int,
    crepe_model: str,
    step_size_ms: int,
    confidence_threshold: float,
    tolerance_cents: float,
    min_stable_ms: float,
    smoothing_window: int,
) -> np.ndarray:
    """
    Run the shared pipeline on one track's audio and return its swara
    sequence as an array of 12-way class indices, shape (num_segments,).

    Uses the track's own annotated tonic as shruthi -- Saraga concerts are
    each sung against a single fixed tonic, same assumption the adherence
    CLI makes for a user-supplied shruthi.
    """
    clip = load_audio(track.audio_path, sample_rate=sample_rate)
    contour = extract_pitch(clip, model=crepe_model, step_size_ms=step_size_ms)
    cents_contour = to_cents_contour(
        contour,
        shruthi_hz=track.tonic_hz,
        confidence_threshold=confidence_threshold,
        octave_fold=True,
    )
    segments = segment_swaras(
        cents_contour,
        tolerance_cents=tolerance_cents,
        min_stable_ms=min_stable_ms,
        smoothing_window=smoothing_window,
    )
    indices = [segment_to_class_index(seg.swara_candidates) for seg in segments]
    return np.asarray(indices, dtype=np.int64)


def _cache_path(cache_dir: Path, track_id: str) -> Path:
    # track_id may contain path separators (it's a relative dir path);
    # flatten to a safe filename.
    safe_name = track_id.replace("/", "__").replace("\\", "__")
    return cache_dir / f"{safe_name}.npz"


def load_or_compute_sequence(
    track: TrackInfo,
    cache_dir: Path,
    pipeline_kwargs: dict,
) -> np.ndarray:
    """
    Return the cached swara sequence for `track`, computing and caching it
    on first access. Regenerating from audio (CREPE inference) on every
    epoch would dominate training wall-clock time for no benefit -- the
    sequence is a deterministic function of the audio and config.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir, track.track_id)

    if path.exists():
        return np.load(path)["swara_indices"]

    sequence = compute_swara_sequence(track, **pipeline_kwargs)
    np.savez(path, swara_indices=sequence)
    return sequence


def build_mvp_raga_list(tracks: Sequence[TrackInfo], top_k: int) -> List[str]:
    """
    Select the `top_k` most-represented ragas by track count. Data-driven,
    not a theory-based pick -- see design log for why.
    """
    counts = raga_counts(tracks)
    return [name for name, _ in counts.most_common(top_k)]


def stratified_split(
    tracks: Sequence[TrackInfo],
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    seed: int = 42,
) -> Tuple[List[TrackInfo], List[TrackInfo], List[TrackInfo]]:
    """
    Stratified train/val/test split by raga, so each split has
    (approximately) the same per-raga proportions as the full set.

    Raises:
        DatasetError: if any raga has fewer than 3 tracks (can't place at
            least one in each of train/val/test).
    """
    test_frac = 1.0 - train_frac - val_frac
    if test_frac <= 0:
        raise DatasetError(
            f"train_frac + val_frac must be < 1.0, got {train_frac + val_frac}"
        )

    by_raga: Dict[str, List[TrackInfo]] = {}
    for t in tracks:
        by_raga.setdefault(t.raga, []).append(t)

    for raga, group in by_raga.items():
        if len(group) < 3:
            raise DatasetError(
                f"Raga {raga!r} has only {len(group)} track(s); need >= 3 to "
                f"place at least one in each of train/val/test."
            )

    rng = np.random.default_rng(seed)
    train: List[TrackInfo] = []
    val: List[TrackInfo] = []
    test: List[TrackInfo] = []

    for raga in sorted(by_raga.keys()):
        group = list(by_raga[raga])
        rng.shuffle(group)
        n = len(group)
        n_train = max(1, round(n * train_frac))
        n_val = max(1, round(n * val_frac))
        # Clamp so train+val leaves at least one for test.
        n_train = min(n_train, n - 2)
        n_val = min(n_val, n - n_train - 1)

        train.extend(group[:n_train])
        val.extend(group[n_train:n_train + n_val])
        test.extend(group[n_train + n_val:])

    return train, val, test


class SaragaSwaraDataset(Dataset):
    """
    PyTorch Dataset over Saraga tracks, yielding (swara_index_tensor, label).

    Args:
        tracks: TrackInfo list for this split.
        raga_to_label: mapping from raga name to integer class index,
            shared across train/val/test so label ids are consistent
            (this is why it's passed in rather than derived per-split).
        cache_dir: where computed swara sequences are cached as .npz.
        pipeline_kwargs: forwarded to `compute_swara_sequence` on cache miss.
    """

    def __init__(
        self,
        tracks: Sequence[TrackInfo],
        raga_to_label: Dict[str, int],
        cache_dir: Path,
        pipeline_kwargs: dict,
    ) -> None:
        unknown = {t.raga for t in tracks} - set(raga_to_label.keys())
        if unknown:
            raise DatasetError(f"Tracks reference ragas not in raga_to_label: {unknown}")

        self._tracks = list(tracks)
        self._raga_to_label = dict(raga_to_label)
        self._cache_dir = cache_dir
        self._pipeline_kwargs = pipeline_kwargs

    def __len__(self) -> int:
        return len(self._tracks)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        track = self._tracks[idx]
        sequence = load_or_compute_sequence(track, self._cache_dir, self._pipeline_kwargs)
        if sequence.shape[0] == 0:
            raise DatasetError(
                f"Track {track.track_id} produced zero swara segments; "
                f"exclude it from the manifest before training."
            )
        return torch.from_numpy(sequence).long(), self._raga_to_label[track.raga]


def collate_fn(
    batch: List[Tuple[torch.Tensor, int]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pad a batch of variable-length swara-index sequences.

    Returns:
        padded: (batch, max_seq_len) long tensor, 0-padded (padding index
            0 collides with the 'S' class, but the model is told the true
            lengths and must use pack_padded_sequence to ignore padding --
            never learns from padded steps regardless of value).
        lengths: (batch,) long tensor of true sequence lengths, required by
            pack_padded_sequence.
        labels: (batch,) long tensor of raga labels.
    """
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in sequences], dtype=torch.long)
    padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return padded, lengths, labels_tensor
