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

Windowing (added after running against real data, not in the original plan):
    Saraga has as few as 3-7 usable tracks for even the most-represented
    ragas. One example per track leaves too little signal for an honest
    train/val/test split -- a handful of tracks per class means a test set
    of ~1 example per class, which produces an F1 number that means nothing.
    `SaragaSwaraDataset` instead slides a fixed-size window over each
    track's swara sequence, producing multiple training examples per track.
    Splitting into train/val/test happens at the TRACK level, before
    windowing (`stratified_split` operates on tracks) -- windows from the
    same concert must never span two different splits, or the model could
    trivially "recognize" a specific performance's pitch quirks instead of
    learning the raga.
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

# Minimum tracks a raga needs to place at least one in each of
# train/val/test. Shared between build_mvp_raga_list (must not select a
# raga it's impossible to split) and stratified_split (enforces it).
MIN_TRACKS_PER_RAGA: int = 3

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
        duration_seconds: audio duration, read once from the file header at
            discovery time (not a full decode) so raga selection can rank by
            total audio volume without re-reading every file on each call.
    """
    track_id: str
    audio_path: Path
    tonic_hz: float
    raga: str
    duration_seconds: float


def _read_tonic(ctonic_path: Path) -> float:
    """Saraga ctonic files hold a single float (Hz) as their only content."""
    text = ctonic_path.read_text(encoding="utf-8").strip()
    try:
        return float(text.split()[0])
    except (ValueError, IndexError) as e:
        raise DatasetError(f"Could not parse tonic from {ctonic_path}: {text!r}") from e


def _read_duration_seconds(audio_path: Path) -> float:
    """
    Read audio duration from the file header/metadata, without decoding the
    full waveform -- used only for ranking ragas by total audio volume.
    """
    import librosa

    return float(librosa.get_duration(path=str(audio_path)))


# Labels Saraga uses in the `raaga` metadata field that name a composition
# *form*, not a single raga (a Rāgamālika performance traverses several
# ragas within one piece, with no sub-raga breakdown recorded). Treated the
# same as a missing raga label: there's no single coherent swara set for
# the identifier to learn from a track labeled this way. Confirmed with
# Shruthi rather than silently including or excluding it.
NON_RAGA_LABELS: frozenset = frozenset({"Rāgamālika"})


def _extract_raaga_name(metadata: dict) -> Optional[str]:
    """
    Same parsing rule as scripts/download_saraga.py's summary printer.
    Returns None for both a missing raga field and a known non-raga label
    (see NON_RAGA_LABELS) -- callers treat both as "no usable raga label".
    """
    raaga = metadata.get("raaga")
    if not raaga or not isinstance(raaga, list):
        return None
    first = raaga[0]
    if isinstance(first, dict):
        name = first.get("name")
    elif isinstance(first, str):
        name = first
    else:
        return None
    return None if name in NON_RAGA_LABELS else name


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

    # __MACOSX/ holds AppleDouble resource-fork sidecar files (._*.json) --
    # binary garbage from macOS zip tooling, not real track metadata. They
    # fail to decode as UTF-8/JSON, so they'd otherwise just fall into the
    # "skipped" bucket below, but filtering them out up front keeps the
    # skip count meaningful (real incomplete tracks vs. zip artifacts).
    metadata_paths = sorted(
        p for p in saraga_root.rglob("*.json") if "__MACOSX" not in p.parts
    )

    for metadata_path in metadata_paths:
        track_dir = metadata_path.parent
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
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

        try:
            duration_seconds = _read_duration_seconds(chosen_audio)
        except Exception:
            # Corrupt/unreadable audio header -- exclude rather than crash
            # the whole scan over one bad file.
            skipped += 1
            continue

        track_id = str(track_dir.relative_to(saraga_root))
        tracks.append(
            TrackInfo(
                track_id=track_id,
                audio_path=chosen_audio,
                tonic_hz=tonic_hz,
                raga=raga,
                duration_seconds=duration_seconds,
            )
        )

    if skipped:
        print(f"discover_tracks: skipped {skipped} incomplete/ambiguous track directories")

    return tracks


def raga_counts(tracks: Sequence[TrackInfo]) -> Counter:
    """Per-raga track counts, most-common first when iterated via .most_common()."""
    return Counter(t.raga for t in tracks)


def raga_durations(tracks: Sequence[TrackInfo]) -> Counter:
    """
    Total audio duration (seconds) per raga.

    Preferred over `raga_counts` for MVP raga selection: Saraga concerts
    range from a few minutes to well over an hour, so a raga with many short
    tracks can have far less actual training signal than one with fewer,
    longer tracks. Duration is measured directly rather than approximated
    from track count.
    """
    durations: Counter = Counter()
    for t in tracks:
        durations[t.raga] += t.duration_seconds
    return durations


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


def build_mvp_raga_list(tracks: Sequence[TrackInfo], top_k: int, by: str = "duration") -> List[str]:
    """
    Select the `top_k` most-represented ragas. Data-driven, not a
    theory-based pick -- see design log for why.

    `by="duration"` (default) ranks by total audio duration -- see
    `raga_durations` for why this beats track count as the selection metric
    on this dataset. `by="count"` ranks by track count instead; kept for
    comparison, not the default.

    Ragas with fewer than `MIN_TRACKS_PER_RAGA` tracks are excluded before
    ranking, regardless of `by`. Without this, duration-based ranking can
    select a raga with a single very long track (lots of audio, zero split
    diversity) that `stratified_split` would immediately reject -- found
    running against the real corpus, where a single 60-minute track
    outranked ragas with 3x the track count on duration alone.
    """
    if by == "duration":
        weights = raga_durations(tracks)
    elif by == "count":
        weights = raga_counts(tracks)
    else:
        raise DatasetError(f"by must be 'duration' or 'count', got {by!r}")

    counts = raga_counts(tracks)
    eligible = {name: w for name, w in weights.items() if counts[name] >= MIN_TRACKS_PER_RAGA}
    return [name for name, _ in Counter(eligible).most_common(top_k)]


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
        DatasetError: if any raga has fewer than MIN_TRACKS_PER_RAGA tracks
            (can't place at least one in each of train/val/test).
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
        if len(group) < MIN_TRACKS_PER_RAGA:
            raise DatasetError(
                f"Raga {raga!r} has only {len(group)} track(s); need >= "
                f"{MIN_TRACKS_PER_RAGA} to place at least one in each of "
                f"train/val/test."
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


def _window_bounds(seq_len: int, window_size: int, stride: int) -> List[Tuple[int, int]]:
    """
    Compute (start, end) index pairs for sliding windows over a sequence of
    length `seq_len`.

    Sequences no longer than `window_size` get a single window covering the
    whole thing rather than being dropped -- with Saraga concerts this
    short (some pieces are only a couple of minutes), discarding them would
    waste already-scarce per-raga signal. Longer sequences get overlapping
    windows at `stride`, with a final window anchored to the tail so the
    last `window_size` segments are never excluded just because they don't
    land on a stride boundary.
    """
    if seq_len == 0:
        return []
    if seq_len <= window_size:
        return [(0, seq_len)]

    bounds: List[Tuple[int, int]] = []
    start = 0
    while start + window_size <= seq_len:
        bounds.append((start, start + window_size))
        start += stride
    if bounds[-1][1] < seq_len:
        bounds.append((seq_len - window_size, seq_len))
    return bounds


class SaragaSwaraDataset(Dataset):
    """
    PyTorch Dataset over windowed Saraga swara sequences, yielding
    (swara_index_tensor, label).

    Each track's cached/computed swara sequence is sliced into overlapping
    fixed-length windows (see module docstring: "Windowing"), and each
    window is a separate example. Track-level train/val/test separation
    must happen *before* constructing this class (via `stratified_split`
    on `tracks`) -- windowing here only ever splits a track's own sequence
    into pieces, so no window can leak across the split boundary its parent
    track was assigned to.

    Args:
        tracks: TrackInfo list for this split.
        raga_to_label: mapping from raga name to integer class index,
            shared across train/val/test so label ids are consistent
            (this is why it's passed in rather than derived per-split).
        cache_dir: where computed swara sequences are cached as .npz.
        pipeline_kwargs: forwarded to `compute_swara_sequence` on cache miss.
        window_size: max swara segments per training example.
        window_stride: hop between window starts for tracks longer than
            `window_size`. Must be positive.
    """

    def __init__(
        self,
        tracks: Sequence[TrackInfo],
        raga_to_label: Dict[str, int],
        cache_dir: Path,
        pipeline_kwargs: dict,
        window_size: int,
        window_stride: int,
    ) -> None:
        unknown = {t.raga for t in tracks} - set(raga_to_label.keys())
        if unknown:
            raise DatasetError(f"Tracks reference ragas not in raga_to_label: {unknown}")
        if window_stride <= 0:
            raise DatasetError(f"window_stride must be positive, got {window_stride}")
        if window_size <= 0:
            raise DatasetError(f"window_size must be positive, got {window_size}")

        self._tracks = list(tracks)
        self._raga_to_label = dict(raga_to_label)

        # Sequences are resolved (cache hit or computed) once at construction
        # time, not per __getitem__ call, so __len__ can be known up front
        # and repeated epochs never re-touch the cache lookup per window.
        self._sequences: List[np.ndarray] = [
            load_or_compute_sequence(t, cache_dir, pipeline_kwargs) for t in self._tracks
        ]

        self._windows: List[Tuple[int, int, int]] = []  # (track_idx, start, end)
        for track_idx, sequence in enumerate(self._sequences):
            for start, end in _window_bounds(len(sequence), window_size, window_stride):
                self._windows.append((track_idx, start, end))

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        track_idx, start, end = self._windows[idx]
        window = self._sequences[track_idx][start:end]
        label = self._raga_to_label[self._tracks[track_idx].raga]
        return torch.from_numpy(window).long(), label

    def labels(self) -> List[int]:
        """
        Per-window labels in dataset order. Used for class-weighting the
        training loss against what the model actually trains on -- track
        count alone would understate the weight a raga needs if its tracks
        happen to be short (fewer windows) relative to other classes.
        """
        return [
            self._raga_to_label[self._tracks[track_idx].raga]
            for track_idx, _, _ in self._windows
        ]

    def track_indices(self) -> List[int]:
        """
        Per-window index into the `tracks` list this dataset was built
        from, in dataset order. Lets a caller aggregate window-level
        predictions back to one-per-track (e.g. the baselines' majority
        vote, or the identifier's own mean-softmax aggregation) without
        reaching into this class's internals.
        """
        return [track_idx for track_idx, _, _ in self._windows]


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
