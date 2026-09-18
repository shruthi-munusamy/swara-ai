"""
Training loop for the LSTM raga identifier.

Design choices:
    - AdamW + cosine annealing: standard, well-behaved defaults for a small
      LSTM; no reason to hand-tune a schedule for an MVP-scale dataset.
    - Class-weighted cross-entropy: even after filtering to the 10
      most-represented ragas, Saraga's per-raga track counts aren't equal,
      and an unweighted loss would bias the model toward whichever raga has
      the most concerts.
    - Early stopping (and checkpointing) on validation macro-F1, not
      accuracy or loss: accuracy on an imbalanced multi-class problem can
      look good while the model ignores minority classes entirely; macro-F1
      penalizes that.
    - Deterministic seeding: torch/numpy/random are all seeded so a training
      run can be reproduced exactly (same split, same seed -> same curves),
      which matters when the val macro-F1 is going to be quoted in the README.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader


def set_seed(seed: int) -> None:
    """Seed python's random, numpy, and torch (CPU + all CUDA devices)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_class_weights(labels: Sequence[int], num_classes: int) -> torch.Tensor:
    """
    Inverse-frequency class weights for `nn.CrossEntropyLoss(weight=...)`.

    weight[c] = total_examples / (num_classes * count[c])

    A class with zero examples in `labels` gets a weight as if it had one
    example, rather than dividing by zero -- this shouldn't happen given
    the >= 3-per-raga constraint in `stratified_split`, but a raga entirely
    absent from a particular training batch/subset shouldn't crash the run.
    """
    counts = np.bincount(np.asarray(labels), minlength=num_classes).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


class EarlyStopping:
    """
    Tracks the best value of a monitored metric and signals when to stop.

    Args:
        patience: number of consecutive non-improving epochs allowed before
            `should_stop` becomes True.
        mode: "max" if higher is better (e.g. macro-F1), "min" if lower is
            better (e.g. loss).
    """

    def __init__(self, patience: int, mode: str = "max") -> None:
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got {mode!r}")
        self.patience = patience
        self.mode = mode
        self.best: Optional[float] = None
        self.counter = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        """Record a new observation. Returns True if it's a new best."""
        is_better = self.best is None or (
            value > self.best if self.mode == "max" else value < self.best
        )
        if is_better:
            self.best = value
            self.counter = 0
            return True

        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False


class WandbLogger:
    """
    Thin wrapper so the training loop doesn't need `if enabled:` checks
    scattered through it, and so tests can run with `enabled=False` without
    importing wandb or touching the network at all.
    """

    def __init__(
        self,
        enabled: bool,
        project: str = "swara-ai",
        entity: Optional[str] = None,
        config: Optional[dict] = None,
    ) -> None:
        self.enabled = enabled
        self._wandb = None
        if enabled:
            import wandb  # local import: only required when actually logging

            self._wandb = wandb
            self._wandb.init(project=project, entity=entity, config=config or {})

    def log(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if self.enabled:
            self._wandb.log(metrics, step=step)

    def finish(self) -> None:
        if self.enabled:
            self._wandb.finish()


@dataclass
class TrainingHistory:
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    val_accuracy: List[float] = field(default_factory=list)
    val_macro_f1: List[float] = field(default_factory=list)
    best_epoch: Optional[int] = None
    stopped_early: bool = False


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0
    for padded, lengths, labels in loader:
        padded, labels = padded.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(padded, lengths)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        total_examples += labels.size(0)
    return total_loss / total_examples


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Compute loss, accuracy, and macro-F1 over an entire dataloader."""
    model.eval()
    total_loss = 0.0
    total_examples = 0
    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for padded, lengths, labels in loader:
            padded, labels = padded.to(device), labels.to(device)
            logits = model(padded, lengths)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            total_examples += labels.size(0)
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    return {
        "loss": total_loss / total_examples,
        "accuracy": accuracy_score(all_labels, all_preds),
        "macro_f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
    }


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_labels: Sequence[int],
    num_classes: int,
    num_epochs: int,
    learning_rate: float,
    checkpoint_path: Path,
    device: Optional[torch.device] = None,
    early_stopping_patience: int = 7,
    wandb_logger: Optional[WandbLogger] = None,
    seed: int = 42,
) -> TrainingHistory:
    """
    Train `model`, checkpointing the best validation-macro-F1 state dict to
    `checkpoint_path` and stopping early if it stops improving.

    Returns:
        TrainingHistory with per-epoch metrics and which epoch was best.
    """
    set_seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    class_weights = compute_class_weights(train_labels, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    early_stopping = EarlyStopping(patience=early_stopping_patience, mode="max")

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    history = TrainingHistory()

    for epoch in range(num_epochs):
        train_loss = _train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history.train_loss.append(train_loss)
        history.val_loss.append(val_metrics["loss"])
        history.val_accuracy.append(val_metrics["accuracy"])
        history.val_macro_f1.append(val_metrics["macro_f1"])

        print(
            f"epoch {epoch + 1}/{num_epochs}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_metrics['loss']:.4f}  "
            f"val_acc={val_metrics['accuracy']:.4f}  "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        if wandb_logger is not None:
            wandb_logger.log(
                {"train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}},
                step=epoch,
            )

        is_best = early_stopping.step(val_metrics["macro_f1"])
        if is_best:
            history.best_epoch = epoch
            torch.save(model.state_dict(), checkpoint_path)

        if early_stopping.should_stop:
            print(f"Early stopping at epoch {epoch + 1} (no improvement in {early_stopping_patience} epochs)")
            history.stopped_early = True
            break

    if wandb_logger is not None:
        wandb_logger.finish()

    return history
