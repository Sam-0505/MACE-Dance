# Musicality Evaluation

Music-motion alignment metrics for the Motion Expert, validated on a small ImperialDance subset.
Everything runs in `Expert-Motion/.venv` on 3D joints at 30 fps.

| Metric | What it asks | Music signal | Motion signal |
|---|---|---|---|
| **BAS** | Does each music beat have a kinematic beat nearby? | beat one-hot (feature dim 34) | local minima of smoothed joint speed |
| **VBCS** | Does each motion accent land near a music beat? | beat one-hot | local maxima of smoothed joint speed |
| **ABHS** | What fraction of music beats get a motion accent within τ? | beat one-hot | local maxima of smoothed joint speed |
| **AAS** | Do stronger musical onsets get stronger kinetic responses? | onset envelope (dim 0) | mean joint acceleration |
| **EAS** | Does motion energy follow musical loudness? | RMS | mean squared joint speed |

`*_max` versions take the best Pearson r over ±0.5 s of lag, and `*_lag` reports that lag (positive = motion follows music).
Every score is also computed against **mismatched music** (from another song or choreography). A metric only says something about musicality if matched scores beat that chance level.

## Settings

- **BAS** is the authors' code (`Expert-Motion/model/metric.py`) copied unchanged: loop over music beats, Gaussian smoothing 5, `exp(-d²/(2·9))` with d in frames. The only change is that `get_db` no longer hard-codes 24 joints.
- **VBCS/ABHS** follow TMD-Bench (arXiv 2605.01809) Eqs. 1–3. The paper gives no σ, τ or smoothing width, so this code uses the BAS values: smoothing 5, σ = τ = 3 frames (0.1 s).
- **Music features** come from `audio_features.py`, a rebuild of EDGE's `baseline` extractor (SR = 30·512). On the shipped `baseline_feats` it reproduces the envelope at r = 0.9999 and nearly all beats. A few slices differ in beat placement because newer librosa beat tracking behaves differently. MACE evaluation uses the shipped features; new audio uses the rebuild.

## MACE-Dance

```bash
cd Expert-Motion
WANDB_MODE=offline .venv/bin/python test.py --batch_size=128 --feature_type baseline   # all 84 test clips, ~25 s
.venv/bin/python ../Evaluation-Musicality/eval_mace.py --pred_dir eval/train_exp2/0
```

Writes `results/mace.json`.

The 84 test slices are 34 s long and start every 17 s, so consecutive slices of a song overlap by half. Run significance tests on one value per song (19 songs), not per slice. With per-song tests (2026-10-09, one diffusion sample per slice), GT EAS beats other-song music in 15/19 songs (p = 0.04) and MACE EAS does not (9/19, p = 0.54). No beat metric beats other-song music for either GT or MACE.

## ImperialDance subset

The four choreographies are listed in `imperial_subset.py`: K-pop ch0 and ch4, Urban ch0, Hip-Hop ch0. The README's music for K-pop ch2/ch3 and for the multi-song Bilibili video is no longer online.

1. Download only these files from the dataset's [Google Drive](https://drive.google.com/drive/folders/1kSndi7ZIljpue_EzYXOpxZenHi52nujL) into `data/imperial/motion/`. There are 12 in total: `dance_level_{e,i,b}_{k_ch0,k_ch4,u_ch0,h_ch0}.npy`.
2. Fetch the music (padded ±10 s around the README range). This needs `yt-dlp` and `ffmpeg`:
   ```bash
   python fetch_imperial_audio.py --ffmpeg $(which ffmpeg)
   ```
3. Run:
   ```bash
   ../Expert-Motion/.venv/bin/python eval_imperial.py
   ```

### Status (2026-10-09): results not usable yet. Music/motion alignment cannot be established

- **Clip lengths don't match the README.** The arrays are not 10 s. They are 12 s (k_ch0), 17 s (k_ch4, u_ch0) and 15 s (h_ch0) at 100 fps, against README music ranges of 15, 12, 15 and 15 s. Some repeat counts are 97 or 101, not 100.
- **The offset search can't find where the motion starts.** The script estimates where each motion starts inside the music window, choosing on even-indexed repeats and scoring on odd ones. The best offsets peak only about 2.6–3.1 SD above the curve's average, which is roughly what chance gives over ~600 offsets. They also disagree with the README starts (for example, h_ch0 lands at 8 s instead of 0 s).
- **Motion doesn't identify its own music.** Scanning each choreography's motion against all four music windows, the motion fits its own music best in only 1 of 4 cases.
- **Repeats are offset from each other** by a median of 0.17–0.4 s, more for beginners. This shift is larger than the 0.1 s beat tolerance, and the data can't say whether it is dancer timing error or recording start time.

**Hip-Hop ch0 only, README start 0 s** (`eval_imperial.py --choreos h_ch0 --trim 2`): every recording begins with ~2 s of large start-up movement at all levels. That movement produced strongly negative EAS/AAS until it was trimmed. After trimming, no level scores higher with the music at 0 s than with the same song shifted 2–8 s or with other songs. Expert > Beginner on BAS (p = 1e-4) holds just as strongly with the wrong music, so it reflects a property of the motion itself, not of musicality. The song's loudness is nearly constant (a practice groove), so EAS has little to track.

Until the exact audio and its start times are known (ask the dataset authors, or align by hand against the source dance videos), beat-level and correlation metrics on this data measure alignment error, not dancer skill.

**Caveat for later:** the dataset has only 5 dancers across the three levels, each repeating a choreography about 100 times. The p-values compare distributions of repeats, not populations of dancers, so read them as effect direction, not as generalization.

### EAS on short clips vs. MACE (2026-10-09): inconclusive

Choreographies: u_ch1, j_ch3, h_ch1, chosen because each source clip is the whole piece of music. MACE dances were generated on the same music (`gen_mace_imperial.py`, 40 samples), then scored with `eval_imperial_eas.py`: one common 11.8 s window, the music offset searched within ±2 s for every group, the offset picked on half the dances, and EAS reported on the other half.

Result: the control (dances paired with another song, given the same search) ranges from −0.05 to +0.59 depending on group and song. Matched scores are not consistently above it, and Expert does not beat Beginner on u_ch1. With only ~12 s of 0.5 s-smoothed signal, choosing the best of ~120 offsets produces large correlations by chance, so these clips cannot separate expertise levels or humans from MACE. h_ch1's loudness is nearly flat (variation 0.06), so EAS has nothing to track there. A first version padded the music with silence, and the music/silence step inflated every score; that padding was removed.

### Multi-seed check (2026-10-09)

10 Motion Expert samples per test slice (`gen_mace_test.py`, then `eas_multiseed.py`), analysed at the song level (19 songs):

| | own-music EAS | other-song EAS | songs own > other | Wilcoxon p |
|---|---|---|---|---|
| GT | 0.184 | 0.050 | 15/19 | 0.040 |
| MACE (mean of 10 seeds) | 0.061 | 0.050 | 8/19 | 0.77 |

- **GT vs MACE:** GT is higher in 15/19 songs (p = 0.026).
- **No individual seed shows an own-music advantage:** own > other in 7–12 of 19 songs, with p from 0.28 to 1.0. MACE's mean EAS per seed ranges from 0.018 to 0.070.
- **Sanity check:** seed-0 MACE BAS is 0.234, against 0.235 from the authors' `test.py`.

### Multi-component analysis (2026-10-09, `multicomponent.py`, design in `docs/musicality_new.docx`)

19 songs, 10 MACE seeds. Each dance is scored against its own song and against 3 wrong songs matched on tempo (same genre preferred), allowing up to 300 ms of lag. Benjamini–Hochberg FDR correction across the 5 components.

| component | Δ real [95% CI] | Δ MACE [95% CI] | G = real − MACE | G p / q | songs Δ>0 (real / MACE) |
|---|---|---|---|---|---|
| loudness ↔ body energy | +.165 [.030, .286] | +.004 [−.024, .037] | +.161 | .016 / .020 | 15 / 7 |
| percussive ↔ upper-body accel. | +.120 [.049, .193] | +.012 [−.001, .025] | +.108 | .004 / .007 | 15 / 12 |
| low-band flux ↔ lower-body speed | +.076 [.036, .115] | −.003 [−.017, .011] | +.079 | <.001 / .002 | 17 / 8 |
| high-band flux ↔ head/hand speed | +.095 [.035, .157] | +.015 [−.002, .032] | +.080 | .004 / .007 | 15 / 13 |
| pulse clarity (approximation) | −.116 [−.213, −.028] | −.006 [−.049, .039] | −.110 | .055 / .055 | 6 / 7 |

- **The four dynamics components** all show the same pattern: real dancers respond more to their own song than to a matched wrong one, and MACE does not.
- **They overlap strongly.** The music features correlate with each other (percussive–low flux r = .74, loudness–percussive r = .62), and so do the motion features. Treat the four as largely one shared intensity effect, not four independent findings.
- **Pulse clarity is reversed even for real dance.** Following the doc's §9 decision rule, drop or redesign it.
- **BAS does not separate own beats from matched wrong-song beats,** for real dance or for MACE (H3).
- **Tempo/genre matching is coarse:** only 14 of the 57 matched pairs share a genre, and the median tempo gap is 5.3 BPM.
