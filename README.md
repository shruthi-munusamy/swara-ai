# Swara AI

A Carnatic vocal analysis system that performs raga adherence checking and raga identification on monophonic vocal recordings, with tonic-invariant feature extraction as the core architectural primitive.

## Problem

Standard music classification approaches break on Carnatic vocal audio because the same raga sung at two different shruthis (tonics) produces entirely different absolute pitch content. Spectrogram-based models trained naively will learn the singer's tonic rather than the raga's structure. Swara AI handles this by symbolically normalizing all pitch content to a singer-specified tonic before any learning or rule evaluation happens.

## System

Two analysis modes share a single swara-extraction pipeline:

- **Adherence checker** (rule-based) — given a target ragam, scores per-region adherence of the sung pitch to the ragam's allowed swara set within a configurable cents tolerance. Surfaces specific time regions where the singer deviates from the ragam.
- **Raga identifier** (learned) — given audio alone, an LSTM classifier trained on Saraga predicts the most likely ragam from the extracted swara sequence.

The split is deliberate: ragam swara sets are canonical, so a learned adherence head would overfit to dataset artifacts rather than the underlying theory. Ragam identification, in contrast, is a real sequence-classification problem with non-trivial temporal structure that benefits from a learned model.

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
              │   CREPE / pYIN   │
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
        ┌───────────┐      ┌──────────┐
        │ Adherence │      │   Raga   │
        │  checker  │      │identifier│
        │  (rules)  │      │  (LSTM)  │
        └───────────┘      └──────────┘
```

Tonic-relative cents conversion is the architectural primitive that makes the rest of the system shruthi-invariant. Without it, the same raga at two different tonics is two different problems.

## Components

**Pitch extraction.** CREPE (Kim et al., ICASSP 2018) as primary — deep CNN trained on synthesized + real audio, state-of-the-art for monophonic pitch tracking. pYIN as fallback for TF-free environments.

**Tonic-relative cents.** `cents = 1200 * log2(f / shruthi_hz)`. All downstream components consume cents, not Hz.

**Swara segmentation.** Median filter on the pitch contour, voicing mask from CREPE confidence, quantization to the nearest swarasthana within `tolerance_cents`. Outputs a swara sequence with per-segment confidence and duration.

**Adherence checker.** Rule-based: for each segmented region, check membership of the swara in the target ragam's allowed swara set. Output includes adherence score, list of out-of-ragam regions with timestamps, and confidence weighting.

**Raga identifier.** Bidirectional LSTM over tonic-normalized swara sequences. Trained on Saraga (CompMusic), starting with the 10 most-represented ragas and scaling up as evaluation supports it. Reports per-raga F1, full confusion matrix, and explicit error analysis on allied raga pairs (Mohanam vs Bilahari, Kharaharapriya vs Kapi) where shared swara sets cause confusion that phrase-level features would resolve.

## Evaluation methodology

- **Primary metrics:** per-raga F1 and full confusion matrix on a held-out Saraga split. Accuracy alone is not informative on class-imbalanced data, which Saraga is.
- **Baselines:** SVM on swara histograms, MLP on aggregated swara features, LSTM on swara sequences. Reported in a comparison table with latency.
- **Allied-raga analysis:** confusion concentrated on theoretically-allied pairs is reported as a known limitation with proposed phrase-level fixes (not as a hidden failure).
- **Out-of-distribution test set:** held-out recordings outside Saraga (own vocal recordings) to test distribution shift, not just dataset accuracy.
- **Latency profile:** end-to-end inference latency reported for both the CREPE and pYIN paths.

## Setup

```bash
git clone https://github.com/shruthi-munusamy/swara-ai.git
cd swara-ai
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Status

- [x] Project scaffold
- [x] Ragam definitions (all 72 melakartas)
- [ ] Swarasthana cents lookup
- [ ] Audio loader
- [ ] CREPE wrapper
- [ ] Tonic-relative cents conversion
- [ ] Swara segmenter
- [ ] Adherence checker
- [ ] LSTM raga identifier (training + evaluation)
- [ ] Baseline comparison (SVM, MLP, LSTM)
- [ ] Allied-raga confusion analysis
- [ ] Out-of-distribution evaluation
- [ ] Latency profiling
- [ ] CLI demo

## Dataset

[Saraga (CompMusic, MTG-UPF)](https://mtg.github.io/saraga/) — research-grade Carnatic music dataset with pre-extracted pitch and pre-annotated raga labels.

## References

- Kim, J. W., Salamon, J., Li, P., & Bello, J. P. (2018). CREPE: A Convolutional Representation for Pitch Estimation. *ICASSP*.
- Mauch, M., & Dixon, S. (2014). pYIN: A fundamental frequency estimator using probabilistic threshold distributions. *ICASSP*.
- Srinivasamurthy, A., Koduri, G. K., Gulati, S., & Serra, X. (2014). Corpora for music information research in Indian art music. *ICMC*.