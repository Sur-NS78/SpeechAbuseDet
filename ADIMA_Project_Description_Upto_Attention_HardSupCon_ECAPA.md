# ADIMA — Multilingual Few-Shot Audio Abuse Detection
## Project Description and Research Context
### Scope: ADIMA/base research through the Whisper Attention + Hard-SupCon + ECAPA-controlled notebook

> **Scope:** This document stops at the uploaded notebook `ADIMA_Attention_HardSupCon_ECAPA_LeakageControlled_PreFewShot(4).ipynb`. Later gated-fusion experiments, corrected-pipeline variants after this notebook, SeamlessM4T, XLS-R/WavLM, and other future extensions are intentionally excluded.

---

## 1. Project Overview

**ADIMA (Abuse Detection In Multilingual Audio)** is a multilingual audio-abuse detection project for ten Indic languages. The task is binary classification directly from speech:

- `0` — Non-abusive
- `1` — Abusive

The research setting is few-shot and cross-lingual: only a limited number of labelled examples are available for adaptation, while the system must operate across multiple languages.

The project is motivated by the limitations of a conventional:

```text
Audio → ASR → Text → Abuse classifier
```

pipeline. ASR errors can be especially problematic for abusive/profane words, and a transcript can discard acoustic information. Direct audio modelling therefore provides an alternative.

The current project asks:

> **Can frozen multilingual speech representations be adapted with cross-language supervised contrastive learning and temporal modelling to improve few-shot audio-abuse detection?**

---

## 2. ADIMA Dataset

The original ADIMA dataset contains:

- **11,775 audio utterances**
- approximately **65 hours** of audio
- **6,446 unique users**
- **10 Indic languages**
- binary abusive/non-abusive labels.

Languages:

1. Bengali
2. Bhojpuri
3. Gujarati
4. Haryanvi
5. Hindi
6. Kannada
7. Malayalam
8. Odia
9. Punjabi
10. Tamil

The dataset was created specifically for multilingual audio-domain abuse/profanity detection.

The original ADIMA work reports expert annotation and quality filtering. It also investigates monolingual and cross-lingual zero-shot settings.

---

## 3. Base Research Paper

The main methodological starting point for this project is:

**Sankaran, Aditya Narayan; Farahbakhsh, Reza; Crespi, Noel. “Towards Cross-Lingual Audio Abuse Detection in Low-Resource Settings with Few-Shot Learning.” COLING 2025.**

The paper investigates few-shot cross-lingual abuse detection on ADIMA using pretrained speech representations including Wav2Vec and Whisper.

Its approach integrates the representations with **Model-Agnostic Meta-Learning (MAML)** and evaluates 50-, 100-, 150-, and 200-shot settings across the ten languages.

The important starting point is therefore:

```text
ADIMA
  ↓
pretrained speech representation
  ↓
few-shot adaptation
  ↓
abusive / non-abusive classification
```

The current project preserves the broad dataset/task/few-shot setting but changes the adaptation strategy.

### Base-paper contribution

The base paper establishes that pretrained speech representations can support few-shot cross-lingual abuse detection in low-resource Indic languages.

Its important components are:

- pretrained audio representations,
- Whisper/Wav2Vec,
- few-shot learning,
- MAML adaptation,
- ten-language ADIMA evaluation,
- 50–200 shot experiments,
- feature visualization.

### Current project's change

The current research question becomes:

> **Can the few-shot support set be used to learn a more discriminative speech representation with supervised contrastive learning instead of relying primarily on MAML-style classifier adaptation?**

---

## 4. Research Problem

The project studies:

> **Few-shot cross-lingual audio abuse detection for low-resource Indic languages.**

The main challenges are:

### 4.1 Limited labelled data

Only a small support set is available during each few-shot experiment.

### 4.2 Cross-language variation

The model should learn the abuse/non-abuse distinction rather than simply learning language identity.

Ideally:

```text
Hindi Abuse
       ↘
        common abuse representation
       ↗
Tamil Abuse
```

while opposite labels remain separated.

### 4.3 Audio-level information

The model operates directly on speech rather than depending on a transcript, allowing acoustic and temporal information to remain available.

---

## 5. Few-Shot Protocol

The evaluated shot sizes are:

```text
50
100
150
200
```

Support examples are balanced across language and class.

The project uses the training pool for support-set construction and keeps the test data separate.

Nested support sampling is used so that smaller support settings are prefixes of a maximum-size sampled pool, making comparisons between shot sizes more controlled.

---

## 6. Whisper Backbone

The current notebook uses:

```text
openai/whisper-medium
```

with:

```text
1024-dimensional encoder representation
16-kHz mono audio
```

Whisper remains **frozen**.

This is deliberate because full fine-tuning of a large pretrained speech model with tiny support sets can lead to severe overfitting.

The project therefore learns lightweight task-specific components on top of the pretrained representation.

---

## 7. Representation Pipeline

The earlier representation approach was:

```text
Whisper frame representations
        ↓
frame-wise L2 normalization
        ↓
temporal mean pooling
        ↓
utterance embedding
```

The current notebook changes the temporal pooling mechanism:

```text
Whisper frame representations
        ↓
frame-wise L2 normalization
        ↓
learned temporal attention pooling
        ↓
utterance embedding
```

The motivation is that simple mean pooling gives every valid frame equal importance. An abusive cue may instead occur in a short temporal region.

Attention allows the model to learn:

```text
unimportant frame → low weight
informative frame → high weight
```

---

## 8. Temporal Attention

The notebook uses a lightweight additive attention mechanism.

Conceptually:

```text
h_t
 ↓
Linear
 ↓
Tanh
 ↓
Linear
 ↓
score s_t
 ↓
Softmax
 ↓
weight α_t
```

and:

```text
z = Σ α_t h_t
```

where `h_t` is the frame representation and `z` is the pooled utterance representation.

The attention module is trained using the support data and then reused unchanged for query/test inference.

Temporal sequences are reduced to a limited number of bins before attention for memory efficiency.

The Whisper backbone itself remains frozen.

---

## 9. Acoustic Branch

The project supplements the pretrained speech representation with handcrafted acoustic features.

The controlled acoustic representation includes features such as:

- MFCC means and standard deviations,
- spectral centroid,
- spectral bandwidth,
- spectral rolloff,
- spectral flatness,
- zero-crossing rate,
- pitch/log-F0 statistics,
- RMS/log-energy statistics.

The purpose is to provide explicit signal-level information that may complement Whisper.

The acoustic branch is treated as a controlled auxiliary representation rather than assuming that all acoustic information is beneficial.

---

## 10. Supervised Contrastive Learning

The main methodological extension is **Supervised Contrastive Learning (SupCon)**.

Instead of learning only a binary decision boundary, SupCon shapes the representation space.

For an anchor:

### Positive

A sample with the same abuse label.

### Negative

A sample with the opposite abuse label.

Conceptually:

```text
Abuse ─── Abuse ─── Abuse
      \    |    /
       similar region

Non-Abuse ─ Non-Abuse
```

The objective encourages same-label examples to be close and opposite-label examples to be separated.

---

## 11. Cross-Language Positive Pairs

A key part of the project is cross-language supervision.

For example:

```text
Hindi Abuse ↔ Tamil Abuse
```

can be a positive relationship.

Likewise:

```text
Malayalam Non-Abuse ↔ Bengali Non-Abuse
```

can be positive.

This encourages the representation to capture abuse-related information shared across languages rather than simply clustering samples by language.

The support batches are therefore multilingual and balanced across language/class combinations.

---

## 12. Hard-Negative SupCon

The current notebook extends SupCon with hard-negative weighting.

A **hard negative** is an opposite-class sample that is highly similar to the anchor.

Example:

```text
Anchor: Abuse
Negative: Non-Abuse
Similarity: very high
```

Such a sample is more informative than an obviously different negative.

The model therefore emphasizes difficult opposite-class examples during contrastive learning.

The current implementation should be described precisely as a:

> **custom hard-negative-weighted supervised contrastive learning variant**

rather than as an exact reproduction of a particular published hard-negative algorithm.

---

## 13. Contrastive Projection

The representation is passed through a projection component before the contrastive objective.

Conceptually:

```text
Whisper + attention + acoustic representation
                    ↓
             projection head
                    ↓
          contrastive representation
                    ↓
                SupCon loss
```

The contrastive temperature is configured around:

```text
0.07
```

The projection representation is used for the contrastive objective while the final few-shot classification stage operates on the learned adapted representation.

---

## 14. Few-Shot Adaptation

Each episode follows the general pattern:

```text
TRAIN
  ↓
balanced support sampling
  ↓
learn task-specific representation
  ↓
SupCon / Hard-SupCon adaptation
  ↓
fit classifier
  ↓
evaluate
```

The support set is the labelled adaptation set.

The test set is not used to train the representation.

This is important because the project is intended to measure performance under limited labelled data.

---

## 15. Data Quality Control

The data pipeline removes unusable recordings such as:

- silent files,
- flat/degenerate files,
- mostly-zero recordings,
- invalid/unreadable audio.

In the controlled pipeline represented by the notebook lineage, 52 files were excluded for audio-quality reasons:

| Reason | Count |
|---|---:|
| Silent | 46 |
| Flat | 4 |
| Mostly zero | 2 |
| **Total** | **52** |

Runtime manifests should remain the authoritative source for the exact final sample counts.

---

## 16. Exact Duplicate Detection

A major issue investigated during the project is train/test audio duplication.

Exact file-level duplicates are detected using hashes such as MD5:

```text
TRAIN hash
    vs
TEST hash
```

If the hashes are identical, the files are exact copies.

The leakage-control strategy does **not** remove useful training recordings simply because a duplicate exists. Instead, suspicious test copies are excluded from the primary evaluation.

This keeps the training pool intact.

---

## 17. ECAPA-Based Near-Duplicate Screening

Exact hashes cannot detect:

- time-shifted copies,
- trimmed copies,
- re-encoded copies,
- slightly modified recordings,
- acoustically near-identical recordings.

The project therefore uses **ECAPA speech embeddings** to screen train/test recordings for high audio similarity.

The correct interpretation is:

> **ECAPA is an audio duplicate/near-duplicate screening mechanism, not proof of speaker identity.**

A high similarity score can indicate an overlapping or near-duplicate recording, but it does not establish that two recordings have the same speaker.

---

## 18. Leakage-Controlled Test Protocol

The primary evaluation follows:

```text
TRAIN
  ↓
unchanged

TEST
  ↓
compare against TRAIN with ECAPA
  ↓
flag suspicious high-similarity recordings
  ↓
remove flagged TEST recordings from primary evaluation
```

The relevant screening threshold is:

```text
maximum train ECAPA similarity ≥ 0.85
```

The training pool remains unchanged.

Unscreened recordings are kept separately and are not silently considered clean.

This produces a primary evaluation that is more conservative with respect to potential audio duplication.

---

## 19. Why the Training Pool Is Not Removed

Removing training samples based on similarity would:

- reduce the available few-shot training pool,
- change the experimental problem,
- potentially create data starvation,
- make comparisons with the original setup less direct.

Therefore the chosen strategy is:

> **keep TRAIN intact; control the primary TEST evaluation.**

This is especially important for a few-shot project.

---

## 20. Primary Evaluation Terminology

The primary result should be described as:

> **ECAPA-screened leakage-controlled evaluation**

or:

> **evaluation on the ECAPA-clean test subset**

It should **not** be called:

> speaker-independent evaluation

because actual speaker identities are not being used as ground-truth metadata.

---

## 21. Train/Test and Development Separation

The project follows:

```text
TRAIN
 ├── support samples
 ├── internal query/development data
 └── model selection

TEST
 └── final evaluation
```

Test labels should not determine architecture or hyperparameter selection.

This prevents test-set performance from becoming part of model selection.

---

## 22. Why Attention Is a Controlled Architectural Experiment

The central representation comparison is:

### Mean pooling

```text
Whisper
 ↓
L2 normalization
 ↓
mean pooling
 ↓
Hard-SupCon
 ↓
classifier
```

versus:

### Attention pooling

```text
Whisper
 ↓
L2 normalization
 ↓
learned temporal attention
 ↓
Hard-SupCon
 ↓
classifier
```

The purpose is to test whether learned temporal weighting is more useful than treating all valid speech frames equally.

---

## 23. Complete Current Architecture

```text
                         ADIMA Audio
                              │
                              ▼
                       16-kHz mono
                              │
                              ▼
                    speech/VAD preprocessing
                              │
                              ▼
                    Frozen Whisper-medium
                              │
                              ▼
                  frame-level representations
                              │
                              ▼
                    frame-wise L2 normalization
                              │
                              ▼
                   learned temporal attention
                              │
                              ▼
                    Whisper representation
                              │
                    ┌─────────┴─────────┐
                    │                   │
                    ▼                   ▼
             Whisper branch      acoustic branch
                    │                   │
                    └─────────┬─────────┘
                              ▼
                           fusion
                              │
                              ▼
                     SupCon projection
                              │
                              ▼
                cross-language Hard-SupCon
                              │
                              ▼
                     few-shot classifier
                              │
                              ▼
                   Abusive / Non-Abusive
```

---

## 24. Frozen and Trainable Components

### Frozen

- Whisper-medium speech encoder.

### Trainable/adapted

- temporal attention pooling,
- task-specific projection/adaptation layers,
- contrastive representation components,
- final few-shot classifier.

The goal is to adapt the representation without fine-tuning the entire pretrained speech model.

---

## 25. Evaluation Metrics

The project evaluates:

### Accuracy

Overall percentage of correct predictions.

### Macro-F1

Average F1 over abusive and non-abusive classes.

Macro-F1 is important because accuracy alone can hide differences between the two classes.

### Language-wise metrics

Results are also inspected by language because a single global score can hide weak performance on individual Indic languages.

---

## 26. Results Represented by the Attention + Hard-SupCon Controlled Notebook

The relevant execution reports approximately:

| Shot | Mean Accuracy | SD | Mean Macro-F1 | SD |
|---:|---:|---:|---:|---:|
| 50 | **79.50%** | 4.18 | **78.57%** | 4.70 |
| 100 | **81.08%** | 3.50 | **80.36%** | 3.81 |
| 150 | **81.36%** | 3.64 | **80.45%** | 4.19 |
| 200 | **82.10%** | 4.16 | **81.12%** | 4.59 |

The main trend is:

```text
50-shot   → ~79.5%
100-shot  → ~81.1%
150-shot  → ~81.4%
200-shot  → ~82.1%
```

Thus performance improves as more labelled support data becomes available.

The standard deviations represent variation across the evaluated support/episode seeds.

---

## 27. Correct Interpretation of the Results

The reported score is for the **complete attention + Hard-SupCon pipeline**.

It should not automatically be interpreted as the isolated contribution of attention because the pipeline simultaneously includes:

- Whisper,
- attention pooling,
- acoustic features,
- cross-language SupCon,
- hard-negative weighting,
- few-shot adaptation,
- leakage-controlled evaluation.

To claim that attention itself improves performance, the corresponding mean-pooling control must be evaluated under the same support seeds, shot sizes, data filtering and test subset.

---

## 28. What Has Been Achieved by This Stage

At the end of this notebook stage, the project has combined:

1. ADIMA multilingual audio-abuse detection
2. few-shot learning
3. frozen Whisper-medium representations
4. learned temporal attention
5. handcrafted acoustic features
6. cross-language supervised contrastive learning
7. hard-negative weighting
8. balanced multilingual support sampling
9. nested shot-size evaluation
10. exact duplicate detection
11. ECAPA-based near-duplicate screening
12. leakage-controlled primary evaluation

The project has therefore progressed beyond:

```text
Whisper → classifier
```

to:

```text
multilingual speech representation
        +
temporal modelling
        +
acoustic information
        +
cross-language contrastive adaptation
        +
hard-negative learning
        +
few-shot classification
        +
audio-leakage control
```

---

## 29. Current Research Contribution

A defensible description of the contribution at this stage is:

> **A leakage-controlled few-shot cross-lingual audio-abuse detection pipeline that adapts frozen multilingual Whisper representations using learned temporal attention and cross-language hard-negative supervised contrastive learning, supplemented by handcrafted acoustic features.**

The novelty should be scoped to the **combination and controlled evaluation** rather than claiming that Whisper, attention, SupCon, acoustic features, or ECAPA are individually novel.

---

## 30. Relationship to the Base Paper

| Component | COLING 2025 base paper | Current project |
|---|---|---|
| Dataset | ADIMA | ADIMA |
| Languages | 10 Indic languages | 10 Indic languages |
| Task | Audio abuse detection | Audio abuse detection |
| Few-shot | 50–200 shots | 50–200 shots |
| Pretrained speech models | Wav2Vec / Whisper | Whisper-medium |
| Main adaptation | MAML | Hard-SupCon-based adaptation |
| Temporal pooling | base representation/pooling | learned temporal attention |
| Acoustic branch | not central | controlled acoustic features |
| Contrastive learning | not the main method | cross-language SupCon |
| Hard negatives | not the main method | included |
| Duplicate control | not the main focus | MD5 + ECAPA screening |
| Primary test | base protocol | ECAPA-screened test |
| TRAIN pool | base training split | retained unchanged |

The main methodological transition is:

```text
Base:
pretrained representation + MAML

Current:
frozen Whisper
      +
temporal attention
      +
acoustic features
      +
cross-language Hard-SupCon
      +
few-shot classifier
      +
leakage-controlled evaluation
```

---

## 32. Key References

### ADIMA dataset paper

Gupta, V., Sharon, R., Sawhney, R., & Mukherjee, D. (2022).  
**ADIMA: Abuse Detection In Multilingual Audio.**  
ICASSP 2022, 6172–6176.

https://arxiv.org/abs/2202.07991

https://github.com/ShareChatAI/ADIMA

### Base few-shot paper

Sankaran, A. N., Farahbakhsh, R., & Crespi, N. (2025).  
**Towards Cross-Lingual Audio Abuse Detection in Low-Resource Settings with Few-Shot Learning.**  
Proceedings of the 31st International Conference on Computational Linguistics (COLING 2025), 5558–5569.

https://aclanthology.org/2025.coling-main.373/

https://aclanthology.org/2025.coling-main.373.pdf

https://github.com/callmesanfornow/fsl-audio-abuse

### Supervised Contrastive Learning

Khosla, P., et al. (2020).  
**Supervised Contrastive Learning.**  
Advances in Neural Information Processing Systems.

### Whisper

Radford, A., et al. (2022).  
**Robust Speech Recognition via Large-Scale Weak Supervision.**

