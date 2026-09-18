"""Tests for the identifier training loop."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.identifier.dataset import NUM_SWARA_CLASSES, collate_fn
from src.identifier.model import RagaLSTM
from src.identifier.train import (
    EarlyStopping,
    TrainingHistory,
    WandbLogger,
    compute_class_weights,
    evaluate,
    set_seed,
    train,
)

NUM_RAGAS = 3


class _TinySequenceDataset(Dataset):
    """A small, deterministic, linearly-separable-by-content dataset."""

    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def _make_toy_data():
    # Each raga gets a distinct, repeated swara motif -- trivially learnable.
    sequences = []
    labels = []
    motifs = [
        [0, 1, 2, 0, 1, 2],
        [5, 6, 7, 5, 6, 7],
        [9, 10, 11, 9, 10, 11],
    ]
    for label, motif in enumerate(motifs):
        for _ in range(6):
            sequences.append(torch.tensor(motif, dtype=torch.long))
            labels.append(label)
    return sequences, labels


def _make_loader(sequences, labels, batch_size=4):
    ds = _TinySequenceDataset(sequences, labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


def _make_model():
    return RagaLSTM(
        num_swara_classes=NUM_SWARA_CLASSES,
        num_ragas=NUM_RAGAS,
        embedding_dim=8,
        hidden_size=16,
        num_layers=2,
        dropout=0.1,
    )


# ----- set_seed -----

class TestSetSeed:

    def test_reproducible_torch_randn(self):
        set_seed(42)
        a = torch.randn(5)
        set_seed(42)
        b = torch.randn(5)
        assert torch.equal(a, b)


# ----- compute_class_weights -----

class TestComputeClassWeights:

    def test_balanced_classes_get_equal_weight(self):
        labels = [0, 0, 1, 1, 2, 2]
        weights = compute_class_weights(labels, num_classes=3)
        assert torch.allclose(weights, torch.full((3,), weights[0].item()))

    def test_rare_class_gets_higher_weight(self):
        labels = [0] * 9 + [1] * 1
        weights = compute_class_weights(labels, num_classes=2)
        assert weights[1] > weights[0]

    def test_absent_class_does_not_crash(self):
        labels = [0, 0, 0]
        weights = compute_class_weights(labels, num_classes=2)
        assert torch.isfinite(weights).all()


# ----- EarlyStopping -----

class TestEarlyStopping:

    def test_improvement_resets_counter(self):
        es = EarlyStopping(patience=2, mode="max")
        assert es.step(0.5) is True
        assert es.step(0.6) is True
        assert es.should_stop is False

    def test_stops_after_patience_non_improving_epochs(self):
        es = EarlyStopping(patience=2, mode="max")
        es.step(0.9)
        assert es.step(0.8) is False
        assert es.should_stop is False
        assert es.step(0.8) is False
        assert es.should_stop is True

    def test_min_mode_prefers_lower_values(self):
        es = EarlyStopping(patience=1, mode="min")
        assert es.step(1.0) is True
        assert es.step(0.5) is True
        assert es.step(0.6) is False

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            EarlyStopping(patience=1, mode="sideways")


# ----- WandbLogger -----

class TestWandbLogger:

    def test_disabled_logger_never_touches_wandb(self):
        logger = WandbLogger(enabled=False)
        # Should not raise even though wandb isn't configured/authenticated.
        logger.log({"loss": 1.0}, step=0)
        logger.finish()
        assert logger.enabled is False


# ----- evaluate -----

class TestEvaluate:

    def test_returns_expected_keys(self):
        sequences, labels = _make_toy_data()
        loader = _make_loader(sequences, labels)
        model = _make_model()
        criterion = nn.CrossEntropyLoss()
        metrics = evaluate(model, loader, criterion, torch.device("cpu"))
        assert set(metrics.keys()) == {"loss", "accuracy", "macro_f1"}

    def test_metrics_in_valid_ranges(self):
        sequences, labels = _make_toy_data()
        loader = _make_loader(sequences, labels)
        model = _make_model()
        criterion = nn.CrossEntropyLoss()
        metrics = evaluate(model, loader, criterion, torch.device("cpu"))
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["macro_f1"] <= 1.0
        assert metrics["loss"] >= 0.0


# ----- train (integration) -----

class TestTrainLoop:

    def test_loss_decreases_and_reaches_perfect_f1_on_trivially_separable_data(self, tmp_path):
        sequences, labels = _make_toy_data()
        train_loader = _make_loader(sequences, labels)
        val_loader = _make_loader(sequences, labels)  # same toy data; this is a plumbing test, not a generalization test
        model = _make_model()

        history = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            train_labels=labels,
            num_classes=NUM_RAGAS,
            num_epochs=15,
            learning_rate=1e-2,
            checkpoint_path=tmp_path / "identifier_best.pt",
            device=torch.device("cpu"),
            early_stopping_patience=15,
            seed=0,
        )

        assert isinstance(history, TrainingHistory)
        # The motifs are trivially separable, so macro-F1 saturates at 1.0
        # almost immediately -- the meaningful signal here is that loss keeps
        # dropping (the loop is actually optimizing), not a slow F1 climb.
        assert history.val_loss[-1] < history.val_loss[0]
        assert history.val_macro_f1[-1] == 1.0

    def test_checkpoint_file_written(self, tmp_path):
        sequences, labels = _make_toy_data()
        loader = _make_loader(sequences, labels)
        model = _make_model()
        checkpoint_path = tmp_path / "identifier_best.pt"

        train(
            model=model,
            train_loader=loader,
            val_loader=loader,
            train_labels=labels,
            num_classes=NUM_RAGAS,
            num_epochs=3,
            learning_rate=1e-2,
            checkpoint_path=checkpoint_path,
            device=torch.device("cpu"),
            early_stopping_patience=10,
            seed=0,
        )
        assert checkpoint_path.exists()

    def test_early_stopping_halts_before_num_epochs(self, tmp_path):
        sequences, labels = _make_toy_data()
        loader = _make_loader(sequences, labels)
        model = _make_model()

        history = train(
            model=model,
            train_loader=loader,
            val_loader=loader,
            train_labels=labels,
            num_classes=NUM_RAGAS,
            num_epochs=200,
            learning_rate=1e-2,
            checkpoint_path=tmp_path / "identifier_best.pt",
            device=torch.device("cpu"),
            early_stopping_patience=3,
            seed=0,
        )
        assert history.stopped_early is True
        assert len(history.train_loss) < 200

    def test_same_seed_gives_identical_loss_curve(self, tmp_path):
        # `train()` seeds torch/numpy/random for everything that happens
        # from its own call onward (data order, dropout), but model
        # *construction* happens before train() is even called -- so
        # reproducing a full run (including initial weights) requires
        # seeding before building the model too, same as any real training
        # script would (seed once at the top, then build + train).
        sequences, labels = _make_toy_data()
        loader = _make_loader(sequences, labels)

        kwargs = dict(
            train_loader=loader,
            val_loader=loader,
            train_labels=labels,
            num_classes=NUM_RAGAS,
            num_epochs=5,
            learning_rate=1e-2,
            device=torch.device("cpu"),
            early_stopping_patience=10,
            seed=123,
        )

        set_seed(123)
        model_a = _make_model()
        history_a = train(model=model_a, checkpoint_path=tmp_path / "a.pt", **kwargs)

        set_seed(123)
        model_b = _make_model()
        history_b = train(model=model_b, checkpoint_path=tmp_path / "b.pt", **kwargs)

        assert history_a.train_loss == pytest.approx(history_b.train_loss)
