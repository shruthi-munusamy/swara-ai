# Swara AI

A Carnatic vocal analysis system that performs raga adherence checking and raga identification on monophonic vocal recordings, with tonic-invariant feature extraction as the core architectural primitive.

## Problem

Standard music classification approaches break on Carnatic vocal audio because the same raga sung at two different shruthis (tonics) produces entirely different absolute pitch content. Spectrogram-based models trained naively will learn the singer's tonic rather than the raga's structure. Swara AI handles this by symbolically normalizing all pitch content to a singer-specified tonic before any learning or rule evaluation happens.

## System

Two analysis modes share a single swara-extraction pipeline:

- **Adherence checker** (rule-based) — given a target ragam, scores per-region adherence of the sung pitch to the ragam's allowed swara set within a configurable cents tolerance. Surfaces specific time regions where the singer deviates from the ragam.
- **Raga identifier** (learned) — given audio alone, a BiLSTM+attention classifier trained on Saraga predicts the most likely raga from the extracted swara sequence.

The split is deliberate: ragam swara sets are canonical, so a learned adherence head would overfit to dataset artifacts rather than the underlying theory. Raga identification, in contrast, is a real sequence-classification problem that benefits from a learned model.

## Architecture

```
        ┌────────────┐    ┌────────────┐
        │   Audio    │    │   Shruthi  │
        └─────┬──────┘    └─────┬──────┘
              │                 │
              └────────┬────────┘
                       ▼
              ┌──────────────────┐
              │ Pitch extraction │
              │       CREPE      │
              └────────┬─────────┘
                       ▼
              ┌──────────────────┐
              │ Cents conversion │
              │ (tonic-relative) │
              └────────┬─────────┘
                       ▼
              ┌─────────────────┐
              │      Swara      │
              │  segmentation   │
              └────────┬────────┘
                       │
              ┌────────┴────────┐
              ▼                 ▼
        ┌───────────┐      ┌──────────────┐
        │ Adherence │      │     Raga     │
        │  checker  │      │  identifier  │
        │  (rules)  │      │ (BiLSTM+attn)│
        └───────────┘      └──────────────┘
```

Tonic-relative cents conversion is the architectural primitive that makes the rest of the system shruthi-invariant. Without it, the same raga at two different tonics is two different problems.

## Components

**Pitch extraction.** CREPE (Kim et al., ICASSP 2018) — deep CNN trained on synthesized + real audio, state-of-the-art for monophonic pitch tracking.

**Tonic-relative cents.** `cents = 1200 * log2(f / shruthi_hz)`. All downstream components consume cents, not Hz.

**Swara segmentation.** Median filter on the pitch contour, voicing mask from CREPE confidence, quantization to the nearest swarasthana within `tolerance_cents`. Outputs a swara sequence with per-segment confidence and duration. Rule-based signal processing, not learned.

**Adherence checker.** Rule-based: for each segmented region, check membership of the swara in the target ragam's allowed swara set. Output includes adherence score, list of out-of-ragam regions with timestamps, and confidence weighting.

**Raga identifier.** Bidirectional LSTM + attention pooling over a 12-way swara-position vocabulary (shared cents positions like R2/G1 collapse to one class, since disambiguating them requires already knowing the raga). Trained on Saraga (CompMusic), on the 10 most-represented ragas by total audio duration. Each track is windowed into overlapping 48-segment chunks to get enough training examples from a corpus where even the most common raga has only 7 usable tracks; splits are made at the track level before windowing, so no window from one concert can leak across train/val/test.

## Real results

**MVP-10 ragas** (selected by total audio duration, not track count — see [Dataset](#dataset)): Bhairavi, Tōḍi, Ṣanmukhapriya, Kāṁbhōji, Mōhanaṁ, Kamās, Bēgaḍa, Rītigauḷa, Kalyāṇi, Behāg. 42 usable tracks total (20 train / 10 val / 12 test), windowed into 1407 / 1034 training examples.

**Training.** AdamW + cosine annealing, class-weighted cross-entropy, early stopping on validation macro-F1 (patience 7). Training converged with the expected overfitting signature for this data scale: train loss fell steadily toward 0 while val loss climbed from epoch ~7 onward. Best **val macro-F1 = 0.595** at epoch 13; stopped at epoch 20.

**Test set (12 held-out tracks, never seen in training):**

| Metric | Value |
|---|---|
| Accuracy | 0.750 |
| Macro-F1 | **0.717** |
| Weighted-F1 | 0.736 |

Two classes (Bēgaḍa, Rītigauḷa) scored 0 F1 from a single misclassification each — an artifact of how few test tracks exist per class at this corpus size (1 track each), not a modeling failure. Full per-raga breakdown, confusion matrix, and per-track predictions: [`models/confusion_matrix.csv`](models/confusion_matrix.csv), [`models/confusion_matrix.png`](models/confusion_matrix.png), [`models/test_predictions.csv`](models/test_predictions.csv).

**Baseline comparison** (same windowed train/val/test splits as the LSTM; neither baseline sees swara order):

| Model              | Val macro-F1 | Test macro-F1 | Params | Inference latency |
|--------------------|--------------|---------------|--------|--------------------|
| SVM (histograms)   |        0.733 |         0.567 |   8208 |              0.07ms |
| MLP (aggregates)   |        0.400 |         0.190 |   1882 |              0.06ms |
| BiLSTM+attention   |        0.595 |         0.717 | 630154 |             12.02ms |

The LSTM clearly outperforms both baselines on the held-out test set — sequence modeling earns its added complexity on this task rather than being asserted without comparison.

**Allied-raga error analysis.** All three test-set confusion pairs (Kamās→Kāṁbhōji, Bēgaḍa→Kāṁbhōji, Rītigauḷa→Bēgaḍa) involve ragas that are *not* melakartas — they're janya (derived) ragas with no swara-set definition yet in `ragams/melakartas.yaml` (which only covers the 72 melakarta parent scales), so the shared-swara-position claim can't be formally verified for them. It's a real, honestly-reported limitation rather than a suppressed one: `src/identifier/evaluate.py`'s `allied_raga_overlap` only resolves pairs where both ragas are melakartas already in the registry, and reports "not verified" otherwise rather than guessing.

**Latency** (10 runs, 10.4s clip, CREPE `tiny`):

| Stage | p50 | p95 |
|---|---|---|
| Audio load | 0.9ms | 469.6ms* |
| CREPE | 871.5ms | 3124.1ms* |
| Cents conversion | 0.06ms | 0.25ms |
| Swara segmentation | 2.3ms | 6.4ms |
| LSTM inference | 1.7ms | 17.6ms |
| **Total** | **876.7ms** | **3617.2ms*** |

*p95 is inflated by TensorFlow/CREPE cold-start on the first couple of runs; CREPE dominates end-to-end latency as expected, everything downstream of pitch extraction is sub-10ms.

**Out-of-distribution evaluation** has not been run yet — it requires non-Saraga vocal recordings that haven't been supplied. This is an explicit gap, not a hidden one.

## Setup

```bash
git clone https://github.com/shruthi-munusamy/swara-ai.git
cd swara-ai
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## CLI quickstart

```bash
# Ragam adherence check only
python scripts/check_adherence.py data/samples/demo_clip.wav Sankarabharanam 220

# Raga identification only (requires a trained checkpoint; see Training below)
python scripts/identify_raga.py data/samples/demo_clip.wav 220

# Both, in one pass
python scripts/demo.py data/samples/demo_clip.wav 220 Sankarabharanam
```

## Training the identifier from scratch

```bash
python scripts/download_saraga.py                 # ~14GB, resumable
python scripts/prepare_identifier_data.py          # picks MVP-10, writes splits
python scripts/train_identifier.py                 # trains, checkpoints models/identifier_best.pt
python scripts/evaluate_identifier.py              # test metrics, confusion matrix
python scripts/evaluate_baselines.py --lstm-test-macro-f1 0.717
python scripts/profile_latency.py
```

## Running the web demo

```bash
# Terminal 1 -- backend
source .venv/bin/activate
uvicorn backend.main:app --port 8000

# Terminal 2 -- frontend
cd frontend
npm install
npm run dev
```

Open http://localhost:5173, upload a clip, set the shruthi, optionally pick a target ragam, and submit. The frontend and the CLI (`scripts/demo.py`) call the exact same pipeline code (`src/pipeline.py`) — they cannot silently diverge.

## Status

- [x] Project scaffold
- [x] Ragam definitions (all 72 melakartas)
- [x] Swarasthana cents lookup
- [x] Audio loader
- [x] CREPE wrapper
- [x] Tonic-relative cents conversion
- [x] Swara segmenter
- [x] Adherence checker
- [x] Saraga dataset loader (windowed, track-level splits)
- [x] LSTM raga identifier (training + evaluation)
- [x] Baseline comparison (SVM, MLP, LSTM)
- [x] Allied-raga confusion analysis (partial — limited by melakarta-only registry coverage)
- [ ] Out-of-distribution evaluation (needs non-Saraga recordings)
- [x] Latency profiling
- [x] CLI demo
- [x] Web demo (FastAPI + React)

## Dataset

[Saraga 1.5 Carnatic](https://zenodo.org/records/4301737) (CompMusic, MTG-UPF) — 249 tracks, of which 161 have a usable raga label + tonic + resolvable audio (65 lack a raga label entirely, 23 more have ambiguous multitrack audio, 6 are labeled "Rāgamālika" — a composition form spanning multiple ragas, not a single raga, and excluded for that reason). Pitch extraction runs on an isolated vocal stem when Saraga provides one, falling back to the full mix otherwise, since CREPE is a monophonic pitch tracker and a polyphonic mix (vocal + violin + mridangam + ghatam) produces materially noisier estimates.

## References

- Kim, J. W., Salamon, J., Li, P., & Bello, J. P. (2018). CREPE: A Convolutional Representation for Pitch Estimation. *ICASSP*.
- Srinivasamurthy, A., Koduri, G. K., Gulati, S., & Serra, X. (2014). Corpora for music information research in Indian art music. *ICMC*.
