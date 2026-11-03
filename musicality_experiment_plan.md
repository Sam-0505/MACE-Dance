# Musicality-Aware Dance Generation: Experiment Plan

**Status:** Experiment blueprint  
**Date:** 2026-11-02  
**Primary goal:** Improve song-specific temporal musicality in generated dance beyond beat synchronization.

---

## 1. Research Question

Can we train a dance generator so that the **temporal dynamics of generated motion follow the temporal dynamics of the conditioning song**, rather than only matching beat timing or generic rhythm?

The current evaluation shows a clear gap between real dancers and MACE on song-specific coupling between music intensity and motion.

### Core music-to-motion targets

Based on `musicality_evaluation.pdf`:

| Music signal | Motion signal | Human own-song Δ | MACE own-song Δ |
|---|---:|---:|---:|
| Treble flux | Body energy | +0.152 | +0.019 |
| Percussive energy | Body energy | +0.149 | +0.011 |
| Percussive energy | Upper-body acceleration | +0.114 | +0.002 |
| Loudness | Body energy | +0.124 | +0.005 |

**Priority:** treble → body energy, then percussion → body energy, percussion → upper-body acceleration, and loudness → body energy.

The working hypothesis is:

> Existing generators can produce plausible and beat-aligned dance, but do not sufficiently use the song's continuous intensity structure to control how strongly and sharply the body moves over time.

---

## 2. Main Prior Work

### 2.1 MoMu-Diffusion — NeurIPS 2024

**Role in this project:** methodological reference for cross-modal rhythmic alignment.

MoMu-Diffusion introduces a bidirectional contrastive rhythmic representation objective. It aligns music and motion in learned latent spaces and constructs rhythmically meaningful positive/negative pairs.

**What we borrow**
- Explicit music–motion alignment rather than relying only on conditioning.
- Contrastive formulation as a possible later extension.
- The idea that rhythmic/motion-amplitude structure can be deliberately supervised.

**How our method differs**
- We initially align **interpretable continuous signals**, not only learned embeddings.
- Examples:
  - loudness ↔ body energy
  - percussion ↔ acceleration
  - treble flux ↔ body energy
- We explicitly model fast and slow temporal scales.

Reference: *MoMu-Diffusion: On Learning Long-Term Motion-Music Synchronization and Correspondence*, NeurIPS 2024.

---

### 2.2 SoulNet / SoulDance — ICCV 2025

**Role in this project:** reference for explicit learned music–motion alignment during generation.

SoulNet uses a music–motion retrieval model as an alignment prior, with local and whole-motion alignment objectives.

**What we borrow**
- Explicit alignment supervision inside a dance generation system.
- Local vs. global alignment as motivation for multiscale supervision.
- A strong learned-alignment baseline for comparison.

**How our method differs**
- We define what should align physically:
  - music intensity curves
  - generated motion energy/acceleration curves
- The loss is interpretable at the signal level.

Reference: *Music-Aligned Holistic 3D Dance Generation via Hierarchical Motion Modeling*, ICCV 2025.

---

### 2.3 Beat-It — ECCV 2024

**Role in this project:** closest methodological precedent for explicit property-level supervision.

Beat-It extracts an interpretable beat-related property from generated motion and directly penalizes mismatch to the musical target.

**What we borrow**
- Explicit differentiable supervision between a measurable music property and a measurable motion property.

**How our method differs**
- Beat-It supervises **beat event timing**.
- We supervise **continuous expressive dynamics**.

Conceptually:

```text
Beat-It:
music beat timing <-> motion beat timing

Ours:
music intensity dynamics <-> motion intensity dynamics
```

Reference: *Beat-It: Beat-Synchronized Multi-Condition 3D Dance Generation*, ECCV 2024.

---

## 3. Base Model Decision

## 3.1 MACE-Dance

MACE is the model where the musicality failure was originally observed.

**Update (2026-11-02): the Motion Expert is trainable.** `Expert-Motion/` is the EDGE codebase with a BiMamba decoder, and it contains everything needed except a `train.py` entry point:
- `EDGE.py` `train_loop` and `model/diffusion.py` `p_losses` (reconstruction, velocity, FK and foot losses, Guidance-Free Training via `train_pred`).
- The cached FineDance training set: `data/dataset_backups/train_tensor_dataset.pkl`, 183 songs, 19,892 slices.
- The released checkpoint `train-3750.pt` with model, optimizer and EMA state and the normalizer. Sampling uses the model weights, not the EMA copy.

Verified: resuming from the checkpoint with optimizer state restored, training runs with stable losses. On one RTX 4090 at batch 64: 20.6 GiB, 0.35 s/step, about 1.8 min per epoch.

### Training facts that affect the losses
- **Training windows are 8 s (240 frames) at 30 fps,** sliced every 1 s. The 40 s horizon applies only at inference. A slow-scale (σ = 15) correlation has only about 5–6 independent points per training window.
- **Training audio is not shipped,** but slice *k* of a song is exactly seconds *k* to *k* + 8 of the FineDance release wav (MFCC match at offset 0, r = 0.987–0.999 on 5 slices). Music signals can be computed for every slice.
- The model output is min-max normalized. The musicality loss must unnormalize before FK; the existing FK loss does not.

### Role of MACE in this project
- First implementation backbone, fine-tuned from the released checkpoint.
- Primary motivation for the problem and the released-model baseline.
- Run the full proposed musicality evaluation on it.

**Decision:** MACE (Motion Expert) is the first implementation backbone.

---

## 3.2 EDGE — Second Backbone

EDGE provides a documented trainable music-to-dance diffusion system with continuous motion outputs and a public training pipeline. It shares the Motion Expert's training code, so the same loss carries over directly.

Reference: *EDGE: Editable Dance Generation From Music*, CVPR 2023.

**Decision:**

```text
First trainable backbone: MACE Motion Expert (fine-tuned from train-3750.pt)
Second-backbone validation: EDGE (trained on FineDance), then SoulNet
```

### Fine-tune vs. from-scratch
- **Development:** fine-tune from the released checkpoint. Control = the same fine-tune with λ = 0 for the same steps and learning rate.
- **Confirmation:** if compute allows, train the best configuration and a baseline from scratch (about 4.7 days per run on one 4090 at 3,750 epochs).

---

## 4. Dataset Strategy

### Stage 0 — Pipeline Reproduction

Fine-tune the released Motion Expert checkpoint with λ = 0.

Goal:
- Verify environment.
- Confirm the λ = 0 fine-tune matches the released model on BAS, FID and own-song Δ.
- Confirm evaluation code.
- Confirm gradients and training stability before modifying the objective.

### Stage 1 — Main Training

Fine-tune on **FineDance**, the Motion Expert's own training set.

Why:
- Better suited to the song-specific intensity relationships being studied.
- Already used in the existing musicality evaluation.
- More appropriate than AIST++ basic for this research question.

### Stage 2 — Held-Out Generalization

Keep **Motorica** held out from training.

Use it to test whether improved musicality generalizes to music and choreography outside the training distribution.

### Dataset caution

AIST++ basic is not ideal for song-specific musicality because routines can be reused across songs. The existing evaluation showed that this weakens own-song specificity.

---

## 5. Input Music Signals

All signals should be aligned to the motion frame rate.

The current evaluation uses **30 FPS**.

### 5.1 Loudness

Use RMS loudness:

\[
L(t) = \sqrt{\frac{1}{N}\sum_n x[tH+n]^2}
\]

Interpretation:
- low during quieter passages
- high during louder/full sections

### 5.2 Percussive Energy

Use HPSS to isolate the percussive component and then compute RMS:

\[
P(t) = \mathrm{RMS}(x_{\text{perc}}(t))
\]

Interpretation:
- high around kicks, snares, claps, and dense percussive passages

### 5.3 Treble Flux

Use positive spectral flux over approximately 6.4–12.8 kHz:

\[
T(t) =
\sum_{k \in B}
\max(0, |X_k(t)|-|X_k(t-1)|)
\]

Interpretation:
- tracks new bright attacks such as hi-hats, cymbals, and claps

---

## 6. Motion Signals

Use predicted joint positions from the generated dance.

### 6.1 Body Energy

\[
E(t)
=
\frac{1}{J}
\sum_j
\|p_j(t+1)-p_j(t)\|^2
\]

Interpretation:
- still poses → low
- large, fast movements → high

### 6.2 Upper-Body Acceleration

\[
A(t)
=
\frac{1}{|U|}
\sum_{j \in U}
\|p_j(t+1)-2p_j(t)+p_j(t-1)\|
\]

Upper-body set should include torso, neck/head, shoulders, elbows, wrists, and hands.

Interpretation:
- pops, hits, snaps, and sharp directional changes → high

---

## 7. Proposed Losses

## 7.1 Loss V1 — Direct Correlation Loss

Start with matched music and generated motion only.

For music signal \(a(t)\) and motion signal \(m(t)\):

\[
\mathcal{L}_{corr}(a,m)
=
1-\rho(a,m)
\]

where \(\rho\) is Pearson correlation.

### Four initial components

\[
\mathcal{L}_T
=
1-\rho(T,E)
\]

\[
\mathcal{L}_{PE}
=
1-\rho(P,E)
\]

\[
\mathcal{L}_{PA}
=
1-\rho(P,A)
\]

\[
\mathcal{L}_L
=
1-\rho(L,E)
\]

Combined:

\[
\mathcal{L}_{dyn}
=
w_T\mathcal{L}_T
+
w_{PE}\mathcal{L}_{PE}
+
w_{PA}\mathcal{L}_{PA}
+
w_L\mathcal{L}_L
\]

### Important

Do **not** replace the original EDGE losses.

Use:

\[
\mathcal{L}_{total}
=
\mathcal{L}_{EDGE}
+
\lambda_{dyn}\mathcal{L}_{dyn}
\]

The original motion-quality losses remain necessary to prevent correlation optimization from producing unrealistic or exaggerated motion.

---

## 7.2 Multiscale Dynamic Loss

The current evaluation already finds different behavior at different timescales.

Use Gaussian smoothing at two scales:

- **Fast:** approximately 2 frames ≈ 67 ms
- **Slow:** approximately 15 frames ≈ 0.5 s

For signal pair \(k\) and scale \(s\):

\[
\mathcal{L}_{multi}
=
\sum_k
\sum_s
w_{k,s}
\left[
1-\rho(a_{k,s},m_{k,s})
\right]
\]

### Initial scale priorities

| Relationship | Fast | Slow |
|---|---:|---:|
| Treble → body energy | High | High |
| Percussion → body energy | Medium | High |
| Percussion → upper-body acceleration | High | Medium |
| Loudness → body energy | High | Evaluate |

Treble should initially receive the highest priority because it shows the strongest unique signal in the current evaluation.

---

## 7.3 Loss V2 — Normalized MSE

Use this as an ablation against correlation.

Z-normalize both signals:

\[
\hat a(t)
=
\frac{a(t)-\mu_a}{\sigma_a+\epsilon}
\]

\[
\hat m(t)
=
\frac{m(t)-\mu_m}{\sigma_m+\epsilon}
\]

Then:

\[
\mathcal{L}_{zMSE}
=
\frac{1}{T}
\sum_t
(\hat a(t)-\hat m(t))^2
\]

This provides a signal-level objective structurally closer to Beat-It.

---

## 7.4 Loss V3 — Temporal Contrastive Loss

Add only after the direct positive-only objective works.

Use the correct temporal music segment as the positive and another segment from the **same song** as a negative.

Example:

\[
S(M_t,D_t)
>
S(M_{t+\Delta},D_t)
\]

This teaches the model that motion should follow the correct **moment** of the track, not merely its overall style or energy distribution.

---

## 7.5 Loss V4 — Song-Specific Contrastive Loss

Later, use a different but similar-tempo song as a hard negative.

\[
S(M^+,D)
>
S(M^-,D)
\]

Margin form:

\[
\mathcal{L}_{rank}
=
\max
\left(
0,
\gamma-S(M^+,D)+S(M^-,D)
\right)
\]

This is optional during the first training stage.

The same-tempo wrong-song comparison should initially remain mainly an **evaluation metric**.

---

## 7.6 Optional Residualized Loss

Loudness, percussion, and treble overlap.

Later, isolate the unique contribution of each signal.

Example:

\[
T_{unique}
=
T-\hat T(L,P)
\]

Then optimize:

\[
1-\rho(T_{unique},E)
\]

This should be introduced only after the simpler objective has been validated.

---

## 8. Evaluation Protocol

## 8.1 Primary Metric — Own-Song Advantage

Keep the current own-song metric as the headline evaluation.

For a given music–motion relationship:

\[
\Delta
=
S_{own}
-
S_{wrong}
\]

where \(S_{wrong}\) is computed using several similar-tempo wrong songs.

Why this is useful:

- controls for generic tempo
- controls for generic rhythmic bouncing
- asks whether the dance follows the **specific temporal structure of its actual song**

Evaluate independently for:

1. treble ↔ body energy
2. percussion ↔ body energy
3. percussion ↔ upper-body acceleration
4. loudness ↔ body energy

---

## 8.2 Raw Own-Song Correlation

Always report:

\[
r_{own}
\]

and

\[
r_{wrong}
\]

in addition to \(\Delta\).

This prevents a model from appearing better simply because wrong-song correlation decreases.

---

## 8.3 Best-Lag Metric

The current evaluation searches within approximately ±300 ms.

Keep the maximized correlation but also expose the maximizing lag:

\[
\tau^*
=
\arg\max_{\tau}
\rho(a(t),m(t+\tau))
\]

Report:

\[
|\tau^*|
\]

This distinguishes:
- strong and timely coupling
- strong but delayed coupling

---

## 8.4 Cross-Wavelet Coherence

Use cross-wavelet coherence as an independent multiscale evaluation.

Report at least:

\[
C_{fast}
\]

and

\[
C_{slow}
\]

This is useful because it measures coupling across temporal frequencies rather than only one smoothed correlation.

It also reduces the risk of evaluating solely with the same mathematical objective used during training.

---

## 8.5 Standard Dance Quality Metrics

The new musicality objective must not damage movement realism.

Retain standard metrics such as:

- FID-style motion fidelity
- diversity
- beat alignment / BAS
- physical foot contact / foot skating metrics
- kinematic plausibility
- qualitative human evaluation if feasible

Evaluation should therefore have two independent dimensions:

```text
A. Motion quality / realism
B. Song-specific temporal musicality
```

---

## 9. Baselines

### Required

1. **MACE, released checkpoint**
2. **MACE, fine-tuned with λ = 0** (controls for extra training)
3. **MACE + proposed loss**
4. **Human ground-truth dance**

### Second backbone and strong additional baseline

5. **EDGE** and **EDGE + proposed loss**
6. **SoulNet**

### Methodological references / optional reproduced baselines

6. MoMu-Diffusion
7. Beat-It

---

## 10. Initial Ablation Plan

| ID | Backbone | Musicality objective |
|---|---|---|
| E0 | MACE | none (λ = 0 fine-tune) |
| E1 | MACE | treble → body energy |
| E2 | MACE | percussion → body energy |
| E3 | MACE | percussion → upper-body acceleration |
| E4 | MACE | loudness → body energy |
| E5 | MACE | all four |
| E6 | MACE | all four + multiscale |
| E7 | MACE | multiscale normalized-MSE |
| E8 | MACE | multiscale correlation + temporal negatives |
| E9 | MACE | multiscale correlation + same-tempo negatives |
| E10 | MACE | full loss + residualized music signals |

Do not run the entire matrix initially. A fine-tune takes about 3 h, so E1–E5 can run back to back once E0 and E1 are in place.

---

## 11. Minimum Viable Experiment

Start with the simplest high-value experiment.

### Baseline

```text
MACE fine-tuned with λ = 0
```

### Proposed model

```text
MACE + treble-to-body-energy loss (fine-tuned)
```

Use the slow treble/body-energy term first:

\[
\mathcal{L}
=
\mathcal{L}_{EDGE}
+
\lambda
\left[
1-\rho(T_{slow},E_{slow})
\right]
\]

Then evaluate:

1. treble own-song \(\Delta\)
2. raw \(r_{own}\)
3. raw \(r_{wrong}\)
4. best lag
5. cross-wavelet coherence
6. BAS
7. FID / motion quality
8. diversity
9. foot-contact quality

### Success criterion

The musicality metric improves substantially **without degrading motion quality**.

If successful:

1. add fast treble loss
2. add percussion → energy
3. add percussion → upper-body acceleration
4. add loudness → energy
5. compare correlation vs z-MSE
6. introduce contrastive negatives only afterward

---

## 12. Proposed Experimental Progression

### Phase 1 — Infrastructure

- [x] Resume training from the Motion Expert checkpoint (smoke test, 2026-11-02)
- [x] Locate training slices in the FineDance release audio
- [x] Add `train.py` with optimizer-state resume
- [ ] λ = 0 fine-tune matches the released checkpoint
- [ ] Verify evaluation metrics
- [ ] Confirm predicted motion representation
- [ ] Convert predicted representation to 3D joints
- [ ] Verify gradients through joint conversion

### Phase 2 — Signal Pipeline

- [x] Loudness extractor
- [x] HPSS percussion extractor
- [x] Treble spectral-flux extractor
- [x] Synchronize all signals to 30 FPS (`Expert-Motion/precompute_music_signals.py`, all 19,892 training slices)
- [x] Body-energy and upper-body acceleration on training motion (numpy, `train_window_check.py`)
- [x] Differentiable (torch) body energy and Gaussian filtering for the loss (upper-body acceleration still to add)
- [x] Mask silent slices (113 training slices, mostly song 169)

### Phase 3 — First Loss

- [x] Implement differentiable Pearson correlation (`Expert-Motion/model/musicality.py`; matches the numpy pipeline to 4e-7)
- [x] Add treble → body-energy term (`train.py --music_signals ... --lambda_dyn`)
- [x] Confirm loss magnitude (1 − r ≈ 0.94; real-motion r = .061 reproduced inside the loss)
- [x] Confirm nonzero gradients
- [ ] Tune \(\lambda_{dyn}\)
- [ ] Verify motion does not collapse or over-amplify

**First run (2026-11-02): λ = 0.01 collapses motion quality within 1 epoch.** At the released checkpoint, the gradient of 1 − r is about 490× the gradient of the EDGE losses (one batch), so λ = 0.01 already makes it about 5× the EDGE losses. After 1 epoch: FID_k 945 (λ = 0: 16.6; released: 14.7), PFC 0.72 (0.23; 0.20), velocity power above 5 Hz 24% (7%). The model adds jitter whose amplitude follows the music. Training r barely moved (.059). λ should be set from gradient size, around 1e-4 to 1e-3.

**λ sweep (2026-11-02, 5 epochs each, `Evaluation-Musicality/finetune_eval.py`, 19 test songs × 10 seeds, results in `results/finetune_sweep.json`):**

| λ | train r (x0) | Δ model, trained pair | FID_k | FID_g | PFC | power > 5 Hz |
|---|---|---|---|---|---|---|
| released | — | +.044 | 14.7 | 25.5 | .204 | 5.0% |
| 0 | .041 | +.054 | 16.7 | 32.1 | .292 | 6.4% |
| 1e-4 | .048 | +.039 | 18.1 | 34.5 | .288 | 6.2% |
| 3e-4 | .057 | +.044 | 18.3 | 32.2 | .269 | 5.6% |
| 1e-3 | .078 | +.058 | 26.2 | 38.9 | .291 | 12.3% |

Real dance: Δ +.182, power > 5 Hz 1.6%. The loss raises r on the training-time x0 predictions (above the real .061 at λ = 1e-3) but not the own-song Δ of sampled test dances, on the trained pair or on any held-out pair. λ = 1e-3 starts adding jitter. The λ = 0 control itself drifts from the release (PFC .204 → .292), so the fine-tune recipe (lr 4e-4, batch 64) needs a lower learning rate.

### Phase 4 — FineDance

- [ ] Precompute music signals (T, P, L) for all 19,892 training slices
- [ ] Fine-tune MACE λ = 0 baseline
- [ ] Fine-tune MACE + musicality loss
- [ ] Compare all metrics

### Phase 5 — Generalization

- [ ] Freeze model
- [ ] Evaluate on Motorica
- [ ] Compare with released MACE
- [ ] Compare against human own-song behavior

### Phase 6 — Extensions

- [ ] all four relationships
- [ ] multiscale loss
- [ ] z-MSE ablation
- [ ] temporal negatives
- [ ] same-tempo negatives
- [ ] residualized signal losses
- [ ] EDGE second-backbone validation
- [ ] SoulNet comparison

---

## 13. Expected Paper Story

### Observation

Modern music-to-dance systems can generate plausible and rhythmically aligned movement but fail to reproduce **song-specific expressive intensity dynamics**.

### Evidence

Real dancers show strong song-specific coupling between:

- treble and body energy
- percussion and body energy
- percussion and upper-body acceleration
- loudness and body energy

while MACE shows only weak coupling.

### Existing methodological precedent

- **MoMu-Diffusion:** cross-modal rhythmic contrastive learning
- **SoulNet:** explicit learned music–motion alignment
- **Beat-It:** interpretable property-level alignment loss

### Proposed contribution

> **Interpretable multiscale temporal supervision that directly couples music dynamics to generated motion dynamics.**

### Evaluation contribution

A musicality evaluation beyond beat alignment using:

- own-song advantage
- raw coupling
- timing lag
- multiscale coherence
- conventional motion-quality metrics

---

## 14. Current Recommended Configuration

### Backbone

**MACE Motion Expert**, fine-tuned from `train-3750.pt`

### Training data

**FineDance**

### External evaluation

**Motorica**

### Comparison models

**Released MACE**, **MACE λ = 0 fine-tune**; EDGE as second backbone

### First loss

\[
1-\rho(T_{slow},E_{slow})
\]

### First extension

Fast + slow treble alignment.

### Next signals

1. percussion → body energy
2. percussion → upper-body acceleration
3. loudness → body energy

### Headline evaluation

**Own-song advantage**

### Complementary evaluation

- raw own-song correlation
- best lag
- cross-wavelet coherence
- FID/diversity/BAS/PFC

---

## 15. References

1. **Tseng et al.** *EDGE: Editable Dance Generation From Music.* CVPR 2023.  
   https://openaccess.thecvf.com/content/CVPR2023/html/Tseng_EDGE_Editable_Dance_Generation_From_Music_CVPR_2023_paper.html

2. **MoMu-Diffusion.** *On Learning Long-Term Motion-Music Synchronization and Correspondence.* NeurIPS 2024.  
   https://proceedings.neurips.cc/paper_files/paper/2024/hash/e6d8684a40187a417461a239238137dd-Abstract-Conference.html

3. **Li et al.** *Music-Aligned Holistic 3D Dance Generation via Hierarchical Motion Modeling.* ICCV 2025.  
   https://openaccess.thecvf.com/content/ICCV2025/html/Li_Music-Aligned_Holistic_3D_Dance_Generation_via_Hierarchical_Motion_Modeling_ICCV_2025_paper.html

4. **Beat-It.** ECCV 2024.  
   https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/2909_ECCV_2024_paper.php

5. **MACE-Dance.** Public project/repository and Motion Expert implementation.  
   https://github.com/AMAP-ML/MACE-Dance

6. **Project evaluation source:** `musicality_evaluation.pdf`, 27 Oct 2026.

---

## 16. Immediate Next Step

Implement one clean proof-of-concept:

```text
MACE fine-tuned with λ = 0
vs.
MACE fine-tuned + slow treble-flux -> body-energy correlation loss
```

Before the first run, check on real training motion whether the slow-scale correlation is stable on 8 s windows (see §3.1). If it is not, start with the fast scale (§7.2 rates fast treble "High" as well).

**Result (2026-11-02, `Evaluation-Musicality/train_window_check.py`): slow is usable, start with slow.** On 18,493 real training slices (149 pieces), treble → energy at σ = 15 gives Δ = +.067 [.043, .090], 107/149 pieces positive, with a higher per-window signal-to-noise ratio than σ = 2 (0.111 vs 0.077). Reference for monitoring training: real-dance own-song r on one 8 s window, slow treble → energy, is .061 on average (SD across windows about .25).

Do not add contrastive negatives, residualization, or all four relationships until this simplest experiment demonstrates that direct temporal musicality supervision can improve own-song coupling without damaging motion realism.
