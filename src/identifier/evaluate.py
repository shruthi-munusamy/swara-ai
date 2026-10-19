"""
Evaluation harness for the LSTM raga identifier.

Evaluates at the TRACK level, aggregating per-window softmax predictions
via mean, not at the window level. This matches how the model is actually
used at inference time: a user submits one whole clip, not a 48-segment
window. The training loop's own validation metric (src/identifier/train.py
evaluate()) is a faster per-window proxy used only to drive early stopping;
the numbers here are the ones that go in the README.

Allied-raga analysis only resolves for ragas that are themselves melakartas
in the existing registry (most MVP-10 ragas are janya/derived ragas not yet
represented there -- see design log). Unresolvable pairs are reported as
such rather than guessed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from src.identifier.dataset import SaragaSwaraDataset, TrackInfo


@dataclass(frozen=True)
class TrackPrediction:
    """One track's aggregated prediction: mean softmax across its windows."""
    track_id: str
    true_label: int
    predicted_label: int
    confidence: float
    top3_labels: Tuple[int, ...]
    top3_confidences: Tuple[float, ...]


def predict_tracks(
    model: nn.Module,
    tracks: Sequence[TrackInfo],
    raga_to_label: Dict[str, int],
    cache_dir: Path,
    pipeline_kwargs: dict,
    window_size: int,
    window_stride: int,
    device: torch.device,
) -> List[TrackPrediction]:
    """
    Run the model on every window of every track, average each track's
    window-level softmax outputs, and return one prediction per track.

    Tracks that produce zero windows (e.g. zero swara segments extracted)
    are silently excluded -- there's nothing to predict from.
    """
    model.eval()
    predictions: List[TrackPrediction] = []
    with torch.no_grad():
        for track in tracks:
            ds = SaragaSwaraDataset(
                [track], raga_to_label, cache_dir, pipeline_kwargs, window_size, window_stride
            )
            if len(ds) == 0:
                continue

            probs_sum = None
            true_label = raga_to_label[track.raga]
            for i in range(len(ds)):
                seq, _ = ds[i]
                seq = seq.unsqueeze(0).to(device)
                length = torch.tensor([seq.shape[1]])
                logits = model(seq, length)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                probs_sum = probs if probs_sum is None else probs_sum + probs

            probs_mean = probs_sum / len(ds)
            order = np.argsort(probs_mean)[::-1]
            top3 = order[:3]
            predictions.append(
                TrackPrediction(
                    track_id=track.track_id,
                    true_label=true_label,
                    predicted_label=int(top3[0]),
                    confidence=float(probs_mean[top3[0]]),
                    top3_labels=tuple(int(i) for i in top3),
                    top3_confidences=tuple(float(probs_mean[i]) for i in top3),
                )
            )
    return predictions


@dataclass(frozen=True)
class EvaluationReport:
    predictions: List[TrackPrediction]
    label_names: List[str]
    per_raga_precision: Dict[str, float]
    per_raga_recall: Dict[str, float]
    per_raga_f1: Dict[str, float]
    macro_f1: float
    weighted_f1: float
    accuracy: float
    confusion: np.ndarray  # (num_classes, num_classes), rows=true, cols=predicted


def evaluate_predictions(predictions: Sequence[TrackPrediction], label_names: Sequence[str]) -> EvaluationReport:
    """Compute per-raga precision/recall/F1, macro/weighted F1, and a confusion matrix."""
    label_names = list(label_names)
    labels = list(range(len(label_names)))
    y_true = [p.true_label for p in predictions]
    y_pred = [p.predicted_label for p in predictions]

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    accuracy = float(np.mean([t == p for t, p in zip(y_true, y_pred)])) if y_true else 0.0
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return EvaluationReport(
        predictions=list(predictions),
        label_names=label_names,
        per_raga_precision={label_names[i]: float(precision[i]) for i in labels},
        per_raga_recall={label_names[i]: float(recall[i]) for i in labels},
        per_raga_f1={label_names[i]: float(f1[i]) for i in labels},
        macro_f1=float(macro_f1),
        weighted_f1=float(weighted_f1),
        accuracy=accuracy,
        confusion=cm,
    )


def top_confusion_pairs(report: EvaluationReport, top_k: int = 5) -> List[Tuple[str, str, int]]:
    """Off-diagonal (true_raga, predicted_raga, count) triples, highest count first."""
    pairs = []
    n = len(report.label_names)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            count = int(report.confusion[i, j])
            if count > 0:
                pairs.append((report.label_names[i], report.label_names[j], count))
    pairs.sort(key=lambda t: t[2], reverse=True)
    return pairs[:top_k]


def allied_raga_overlap(true_raga: str, predicted_raga: str) -> str:
    """
    Check whether two confused ragas share swara positions, via the
    melakarta registry.

    Only resolves when both names are melakartas already in
    ragams/melakartas.yaml -- most MVP-10 ragas are janya (derived) ragas
    not represented there. Returns a plain-language verdict rather than
    raising, so an unresolvable pair degrades the report instead of
    crashing it.
    """
    from src.swara.ragams import load_ragams

    registry = load_ragams()
    try:
        r1 = registry.get_by_name(true_raga)
        r2 = registry.get_by_name(predicted_raga)
    except KeyError:
        return "not verified (one or both ragas are not melakartas in the registry)"

    shared = r1.swaras_cents & r2.swaras_cents
    total = r1.swaras_cents | r2.swaras_cents
    overlap_frac = len(shared) / len(total) if total else 0.0
    return f"{len(shared)}/{len(total)} swara positions shared ({overlap_frac:.0%})"
