"""
LSTM raga identifier.

Input: a batch of swara-class-index sequences (see src/identifier/dataset.py
for the 12-way vocabulary) with per-sample lengths for the padded positions.
Output: per-raga logits.

Architecture choice: BiLSTM + attention pooling, not a Transformer.
    With an MVP training set on the order of 100-200 examples per class,
    self-attention's lack of a built-in sequence-order prior would need to be
    learned from very little data and would be prone to overfitting; the
    LSTM's recurrence is a useful inductive bias at this data scale. A
    Transformer only pays off once there's enough data for its added
    flexibility to be learned rather than memorized.

Why attention pooling instead of just the final hidden state: a full
Carnatic phrase's raga-identifying information (specific swara combinations,
gamaka-adjacent transitions) can occur anywhere in the sequence, not just at
the end. Attention lets the model learn which positions matter instead of
being forced through a single final-timestep bottleneck.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class AttentionPooling(nn.Module):
    """
    Additive attention pooling over a sequence of hidden states.

    Learns a single query vector; each timestep's relevance is scored
    against it, padded positions are masked out before the softmax, and the
    output is the attention-weighted sum of hidden states.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.query = nn.Parameter(torch.randn(hidden_dim) * (hidden_dim ** -0.5))

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_dim)
            mask: (batch, seq_len) bool, True at valid (non-padded) positions.

        Returns:
            pooled: (batch, hidden_dim)
            weights: (batch, seq_len) attention weights (0 at padded positions).
        """
        scores = torch.tanh(self.proj(hidden_states)) @ self.query  # (batch, seq_len)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(hidden_states * weights.unsqueeze(-1), dim=1)
        return pooled, weights


class RagaLSTM(nn.Module):
    """
    Bidirectional LSTM over swara sequences for raga classification.

    Input: (batch, seq_len) long tensor of swara-class indices in
        [0, num_swara_classes), plus (batch,) lengths for padded positions.
    Output: (batch, num_ragas) logits.

    Note on padding: the embedding layer does NOT use `padding_idx=0`, even
    though `collate_fn` pads with 0. Index 0 is also the real 'S' swara
    class, so zeroing its embedding row (what `padding_idx` does) would
    prevent the model from ever learning a representation for Sa. Padded
    positions are excluded from the LSTM's recurrence via
    `pack_padded_sequence` and from pooling via the attention mask instead —
    correctness comes from the packing/masking, not from a padding index.
    """

    def __init__(
        self,
        num_swara_classes: int,
        num_ragas: int,
        embedding_dim: int = 32,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_swara_classes, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        pooled_dim = hidden_size * 2  # bidirectional
        self.attention = AttentionPooling(pooled_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(pooled_dim, num_ragas)

    def forward(self, swara_indices: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            swara_indices: (batch, seq_len) long tensor, 0-padded.
            lengths: (batch,) long tensor of true (unpadded) sequence lengths.

        Returns:
            (batch, num_ragas) logits.
        """
        embedded = self.embedding(swara_indices)  # (batch, seq_len, embedding_dim)

        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        hidden_states, _ = pad_packed_sequence(
            packed_out, batch_first=True, total_length=swara_indices.size(1)
        )

        seq_len = swara_indices.size(1)
        positions = torch.arange(seq_len, device=swara_indices.device).unsqueeze(0)
        mask = positions < lengths.to(swara_indices.device).unsqueeze(1)  # (batch, seq_len)

        pooled, _ = self.attention(hidden_states, mask)
        pooled = self.dropout(pooled)
        return self.classifier(pooled)
