"""Tests for the LSTM raga identifier model."""

import torch
import torch.nn as nn

from src.identifier.dataset import NUM_SWARA_CLASSES, collate_fn
from src.identifier.model import RagaLSTM

NUM_RAGAS = 5


def _make_model(**overrides) -> RagaLSTM:
    kwargs = dict(
        num_swara_classes=NUM_SWARA_CLASSES,
        num_ragas=NUM_RAGAS,
        embedding_dim=8,
        hidden_size=16,
        num_layers=2,
        dropout=0.3,
    )
    kwargs.update(overrides)
    return RagaLSTM(**kwargs)


def _random_batch(batch_size=4, max_len=10, seed=0):
    torch.manual_seed(seed)
    lengths = torch.randint(low=2, high=max_len + 1, size=(batch_size,))
    sequences = [
        torch.randint(0, NUM_SWARA_CLASSES, (int(length),)) for length in lengths
    ]
    labels = [0] * batch_size
    padded, lengths, labels_t = collate_fn(list(zip(sequences, labels)))
    return padded, lengths, labels_t


class TestForwardShape:

    def test_output_shape_matches_batch_and_num_ragas(self):
        model = _make_model()
        padded, lengths, _ = _random_batch(batch_size=4, max_len=12)
        logits = model(padded, lengths)
        assert logits.shape == (4, NUM_RAGAS)

    def test_single_example_batch(self):
        model = _make_model()
        padded, lengths, _ = _random_batch(batch_size=1, max_len=6)
        logits = model(padded, lengths)
        assert logits.shape == (1, NUM_RAGAS)

    def test_varying_sequence_lengths_in_same_batch(self):
        model = _make_model()
        sequences = [
            torch.tensor([1, 2, 3, 4, 5]),
            torch.tensor([6]),
            torch.tensor([1, 1]),
        ]
        labels = [0, 1, 2]
        padded, lengths, _ = collate_fn(list(zip(sequences, labels)))
        logits = model(padded, lengths)
        assert logits.shape == (3, NUM_RAGAS)


class TestGradientFlow:

    def test_gradients_reach_all_named_parameters(self):
        model = _make_model()
        padded, lengths, labels = _random_batch(batch_size=4, max_len=8)
        logits = model(padded, lengths)
        loss = nn.functional.cross_entropy(logits, labels)
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"{name} received no gradient"
            assert torch.any(param.grad != 0), f"{name} gradient is all zero"

    def test_class_zero_embedding_receives_gradient(self):
        # Regression guard: padding_idx=0 would zero this row's gradient
        # forever, which would break the real 'S' swara class (also index 0).
        model = _make_model()
        sequences = [torch.tensor([0, 1, 2]), torch.tensor([0, 0])]
        labels = [0, 1]
        padded, lengths, labels_t = collate_fn(list(zip(sequences, labels)))
        logits = model(padded, lengths)
        loss = nn.functional.cross_entropy(logits, labels_t)
        loss.backward()
        assert torch.any(model.embedding.weight.grad[0] != 0)


class TestPaddingInvariance:

    def test_padded_region_content_does_not_affect_output(self):
        model = _make_model()
        model.eval()

        real = torch.tensor([3, 5, 7])
        length = torch.tensor([3])

        seq_a = torch.cat([real, torch.zeros(4, dtype=torch.long)]).unsqueeze(0)
        seq_b = torch.cat([real, torch.full((4,), 11, dtype=torch.long)]).unsqueeze(0)

        with torch.no_grad():
            out_a = model(seq_a, length)
            out_b = model(seq_b, length)

        assert torch.allclose(out_a, out_b, atol=1e-6)


class TestOverfitSanityCheck:

    def test_model_can_overfit_a_single_batch(self):
        torch.manual_seed(0)
        model = _make_model(num_ragas=3, embedding_dim=16, hidden_size=32)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        sequences = [
            torch.tensor([0, 1, 2, 3]),
            torch.tensor([5, 6, 7]),
            torch.tensor([9, 10, 11, 0, 1]),
        ]
        labels = [0, 1, 2]
        padded, lengths, labels_t = collate_fn(list(zip(sequences, labels)))

        losses = []
        for _ in range(200):
            optimizer.zero_grad()
            logits = model(padded, lengths)
            loss = nn.functional.cross_entropy(logits, labels_t)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            final_logits = model(padded, lengths)
            predictions = final_logits.argmax(dim=1)

        assert losses[-1] < 0.05
        assert torch.equal(predictions, labels_t)
