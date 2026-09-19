"""Tests for the Saraga dataset loader and preprocessing pipeline."""

import json
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from src.identifier.dataset import (
    NUM_SWARA_CLASSES,
    DatasetError,
    SaragaSwaraDataset,
    TrackInfo,
    build_mvp_raga_list,
    cents_to_class_index,
    collate_fn,
    compute_swara_sequence,
    discover_tracks,
    load_or_compute_sequence,
    raga_counts,
    segment_to_class_index,
    stratified_split,
)
from src.pitch.extractor import CREPE_SAMPLE_RATE

PIPELINE_KWARGS = dict(
    sample_rate=CREPE_SAMPLE_RATE,
    crepe_model="tiny",
    step_size_ms=10,
    confidence_threshold=0.5,
    tolerance_cents=35.0,
    min_stable_ms=80.0,
    smoothing_window=5,
)


# ----- Class indexing -----

class TestCentsToClassIndex:

    def test_all_12_canonical_positions_map_uniquely(self):
        positions = [i * 100.0 for i in range(12)]
        indices = [cents_to_class_index(c) for c in positions]
        assert indices == list(range(12))

    def test_non_canonical_cents_raises(self):
        with pytest.raises(DatasetError):
            cents_to_class_index(150.0)

    def test_shared_position_swaras_collapse_to_same_index(self):
        # R2 and G1 both sit at 200 cents -- should be indistinguishable.
        assert segment_to_class_index(("G1", "R2")) == segment_to_class_index(("R2",))

    def test_empty_candidates_raises(self):
        with pytest.raises(DatasetError):
            segment_to_class_index(())


# ----- discover_tracks -----

def _write_sine(path: Path, freq_hz: float, duration_s: float, sr: int) -> None:
    n = int(duration_s * sr)
    t = np.arange(n) / sr
    wave = (0.5 * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float32)
    sf.write(str(path), wave, sr)


def _make_track_dir(root: Path, name: str, raga: str, tonic_hz: float = 220.0, complete: bool = True) -> Path:
    track_dir = root / name
    track_dir.mkdir(parents=True)

    (track_dir / f"{name}.json").write_text(
        json.dumps({"raaga": [{"name": raga}], "title": name})
    )
    if complete:
        (track_dir / f"{name}.ctonic.txt").write_text(f"{tonic_hz}\n")
        _write_sine(track_dir / f"{name}.wav", 220.0, 0.5, 16000)
    return track_dir


@pytest.fixture
def saraga_fixture(tmp_path):
    root = tmp_path / "saraga"
    _make_track_dir(root, "track_a", raga="Kalyani", tonic_hz=220.0)
    _make_track_dir(root, "track_b", raga="Kalyani", tonic_hz=145.0)
    _make_track_dir(root, "track_c", raga="Sankarabharanam", tonic_hz=196.0)
    # Incomplete: no ctonic file, no audio -- should be skipped.
    _make_track_dir(root, "track_incomplete", raga="Todi", complete=False)
    return root


class TestDiscoverTracks:

    def test_finds_complete_tracks(self, saraga_fixture):
        tracks = discover_tracks(saraga_fixture)
        track_ids = {t.track_id for t in tracks}
        assert track_ids == {"track_a", "track_b", "track_c"}

    def test_skips_incomplete_tracks(self, saraga_fixture):
        tracks = discover_tracks(saraga_fixture)
        assert "track_incomplete" not in {t.track_id for t in tracks}

    def test_reads_tonic_and_raga(self, saraga_fixture):
        tracks = {t.track_id: t for t in discover_tracks(saraga_fixture)}
        assert tracks["track_a"].tonic_hz == pytest.approx(220.0)
        assert tracks["track_a"].raga == "Kalyani"

    def test_ignores_macosx_resource_fork_files(self, saraga_fixture):
        # Real Saraga zips extract a __MACOSX/ tree of AppleDouble sidecar
        # files (._*.json) that aren't valid UTF-8/JSON. discover_tracks
        # must skip them silently rather than crashing or miscounting.
        macosx_dir = saraga_fixture / "__MACOSX" / "track_a"
        macosx_dir.mkdir(parents=True)
        (macosx_dir / "._track_a.json").write_bytes(b"\x00\x01\xb4\xff not valid json")

        tracks = discover_tracks(saraga_fixture)
        track_ids = {t.track_id for t in tracks}
        assert track_ids == {"track_a", "track_b", "track_c"}

    def test_instrument_stem_excluded_from_mix_candidates(self, tmp_path):
        root = tmp_path / "saraga"
        track_dir = _make_track_dir(root, "track_x", raga="Todi")
        # Add a separated instrument stem alongside the mix; discover_tracks
        # must not mistake it for the mix (no vocal stem here, so it should
        # fall back to the plain mix file).
        _write_sine(track_dir / "track_x.multitrack-violin.wav", 300.0, 0.3, 16000)
        tracks = {t.track_id: t for t in discover_tracks(root)}
        assert tracks["track_x"].audio_path.name == "track_x.wav"

    def test_prefers_isolated_vocal_stem_over_mix(self, tmp_path):
        root = tmp_path / "saraga"
        track_dir = _make_track_dir(root, "track_y", raga="Todi")
        # An isolated vocal channel should win over the polyphonic mix --
        # CREPE is monophonic and the mix is noisier.
        _write_sine(track_dir / "track_y.multitrack-vocal.wav", 300.0, 0.3, 16000)
        tracks = {t.track_id: t for t in discover_tracks(root)}
        assert tracks["track_y"].audio_path.name == "track_y.multitrack-vocal.wav"

    def test_multiple_vocal_stems_is_ambiguous_and_skipped(self, tmp_path):
        root = tmp_path / "saraga"
        track_dir = _make_track_dir(root, "track_z", raga="Todi")
        _write_sine(track_dir / "track_z.multitrack-vocal.wav", 300.0, 0.3, 16000)
        _write_sine(track_dir / "track_z.multitrack-vocal2.wav", 300.0, 0.3, 16000)
        tracks = {t.track_id: t for t in discover_tracks(root)}
        assert "track_z" not in tracks


# ----- raga_counts / build_mvp_raga_list -----

class TestRagaCounts:

    def test_counts_data_driven_not_alphabetical(self):
        tracks = [
            TrackInfo("t1", Path("a"), 220.0, "Todi"),
            TrackInfo("t2", Path("b"), 220.0, "Todi"),
            TrackInfo("t3", Path("c"), 220.0, "Todi"),
            TrackInfo("t4", Path("d"), 220.0, "Kalyani"),
        ]
        top = build_mvp_raga_list(tracks, top_k=1)
        assert top == ["Todi"]

    def test_raga_counts_tally(self):
        tracks = [
            TrackInfo("t1", Path("a"), 220.0, "Todi"),
            TrackInfo("t2", Path("b"), 220.0, "Kalyani"),
        ]
        counts = raga_counts(tracks)
        assert counts["Todi"] == 1
        assert counts["Kalyani"] == 1


# ----- stratified_split -----

def _fake_tracks(raga_sizes: dict) -> list:
    tracks = []
    i = 0
    for raga, n in raga_sizes.items():
        for _ in range(n):
            tracks.append(TrackInfo(f"t{i}", Path(f"p{i}"), 220.0, raga))
            i += 1
    return tracks


class TestStratifiedSplit:

    def test_every_raga_appears_in_every_split(self):
        tracks = _fake_tracks({"Todi": 10, "Kalyani": 10, "Sankarabharanam": 10})
        train, val, test = stratified_split(tracks, seed=42)
        for split in (train, val, test):
            ragas_present = {t.raga for t in split}
            assert ragas_present == {"Todi", "Kalyani", "Sankarabharanam"}

    def test_no_track_duplicated_or_dropped(self):
        tracks = _fake_tracks({"Todi": 10, "Kalyani": 10})
        train, val, test = stratified_split(tracks, seed=42)
        all_ids = [t.track_id for t in train + val + test]
        assert sorted(all_ids) == sorted(t.track_id for t in tracks)
        assert len(set(all_ids)) == len(all_ids)

    def test_split_ratios_approximately_correct(self):
        tracks = _fake_tracks({"Todi": 100})
        train, val, test = stratified_split(tracks, train_frac=0.6, val_frac=0.2, seed=42)
        assert len(train) == pytest.approx(60, abs=2)
        assert len(val) == pytest.approx(20, abs=2)
        assert len(test) == pytest.approx(20, abs=2)

    def test_deterministic_given_seed(self):
        tracks = _fake_tracks({"Todi": 20, "Kalyani": 20})
        split_a = stratified_split(tracks, seed=42)
        split_b = stratified_split(tracks, seed=42)
        assert [t.track_id for t in split_a[0]] == [t.track_id for t in split_b[0]]

    def test_raga_with_too_few_tracks_raises(self):
        tracks = _fake_tracks({"Todi": 2, "Kalyani": 10})
        with pytest.raises(DatasetError):
            stratified_split(tracks, seed=42)


# ----- compute_swara_sequence / caching -----

@pytest.fixture
def sine_track(tmp_path):
    """A short synthetic track: steady Sa (220 Hz tonic == the sung pitch)."""
    audio_path = tmp_path / "sine.wav"
    _write_sine(audio_path, 220.0, duration_s=1.5, sr=CREPE_SAMPLE_RATE)
    return TrackInfo(track_id="sine_track", audio_path=audio_path, tonic_hz=220.0, raga="Sankarabharanam")


class TestComputeSwaraSequence:

    def test_returns_int64_array(self, sine_track):
        seq = compute_swara_sequence(sine_track, **PIPELINE_KWARGS)
        assert seq.dtype == np.int64

    def test_values_within_class_range(self, sine_track):
        seq = compute_swara_sequence(sine_track, **PIPELINE_KWARGS)
        assert np.all((seq >= 0) & (seq < NUM_SWARA_CLASSES))

    def test_steady_sa_produces_class_zero(self, sine_track):
        # 220 Hz sung against a 220 Hz tonic is Sa (0 cents) -> class index 0.
        seq = compute_swara_sequence(sine_track, **PIPELINE_KWARGS)
        assert len(seq) > 0
        assert seq[0] == 0


class TestCaching:

    def test_cache_hit_is_faster_than_compute(self, sine_track, tmp_path):
        cache_dir = tmp_path / "cache"

        start = time.perf_counter()
        first = load_or_compute_sequence(sine_track, cache_dir, PIPELINE_KWARGS)
        uncached_time = time.perf_counter() - start

        start = time.perf_counter()
        second = load_or_compute_sequence(sine_track, cache_dir, PIPELINE_KWARGS)
        cached_time = time.perf_counter() - start

        assert np.array_equal(first, second)
        assert cached_time < uncached_time

    def test_writes_npz_file(self, sine_track, tmp_path):
        cache_dir = tmp_path / "cache"
        load_or_compute_sequence(sine_track, cache_dir, PIPELINE_KWARGS)
        assert (cache_dir / "sine_track.npz").exists()


# ----- SaragaSwaraDataset + collate_fn -----

class TestSaragaSwaraDataset:

    def test_len_matches_track_count(self, sine_track, tmp_path):
        ds = SaragaSwaraDataset([sine_track], {"Sankarabharanam": 0}, tmp_path / "cache", PIPELINE_KWARGS)
        assert len(ds) == 1

    def test_getitem_returns_tensor_and_label(self, sine_track, tmp_path):
        ds = SaragaSwaraDataset([sine_track], {"Sankarabharanam": 0}, tmp_path / "cache", PIPELINE_KWARGS)
        sequence, label = ds[0]
        assert isinstance(sequence, torch.Tensor)
        assert sequence.dtype == torch.long
        assert label == 0

    def test_unknown_raga_raises(self, sine_track, tmp_path):
        with pytest.raises(DatasetError):
            SaragaSwaraDataset([sine_track], {"Todi": 0}, tmp_path / "cache", PIPELINE_KWARGS)

    def test_labels_consistent_across_datasets_sharing_mapping(self, sine_track, tmp_path):
        mapping = {"Sankarabharanam": 3, "Kalyani": 1}
        ds1 = SaragaSwaraDataset([sine_track], mapping, tmp_path / "cache", PIPELINE_KWARGS)
        ds2 = SaragaSwaraDataset([sine_track], mapping, tmp_path / "cache", PIPELINE_KWARGS)
        assert ds1[0][1] == ds2[0][1] == 3


class TestCollateFn:

    def test_pads_to_max_length(self):
        batch = [
            (torch.tensor([1, 2, 3]), 0),
            (torch.tensor([4, 5]), 1),
        ]
        padded, lengths, labels = collate_fn(batch)
        assert padded.shape == (2, 3)
        assert torch.equal(lengths, torch.tensor([3, 2]))
        assert torch.equal(labels, torch.tensor([0, 1]))

    def test_padding_does_not_corrupt_real_values(self):
        batch = [(torch.tensor([7, 8, 9]), 0), (torch.tensor([1]), 1)]
        padded, lengths, _ = collate_fn(batch)
        assert torch.equal(padded[0], torch.tensor([7, 8, 9]))
        assert padded[1, 0] == 1
