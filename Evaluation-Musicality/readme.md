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

### Loudness robustness sweep (2026-10-26, `robustness_loud.py`)

Loudness ↔ body energy only. Each variant changes one setting from the primary pipeline, and every variant is reported. The primary now uses **tempo-first matching**: wrong songs are ranked by relative tempo distance, with half/double tempo allowed and ties within 3% going to the same genre. BPM is computed from the full song (60 / mean inter-beat interval) instead of the old 1800 / median interval in frames, which was quantised and badly off for 063 and 211 (138.5 vs ~117 and ~97). Median tempo distance of the matched pairs is now 3%, maximum 13% (song 198 has no close neighbour). With the old matching (`match_genre_first`) the script reproduces `multicomponent.json` exactly.

| variant | Δ real | Δ MACE | G [95% CI] | G perm p | real own − MACE own (p) |
|---|---|---|---|---|---|
| **primary** | +.111 | +.006 | +.104 [.015, .188] | .034 | +.118 (.019) |
| old genre-first matching | +.165 | +.004 | +.161 [.041, .272] | .016 | +.118 (.019) |
| music: RMS in dB | +.121 | +.005 | +.116 [.025, .202] | .022 | +.119 (.013) |
| music: A-weighted dB | +.110 | +.018 | +.093 [.013, .165] | .035 | +.079 (.071) |
| motion: mean speed | +.128 | −.002 | +.130 [.037, .218] | .013 | +.159 (.005) |
| motion: pelvis-relative | +.100 | +.012 | +.088 [.001, .167] | .056 | +.106 (.027) |
| smoothing σ = 8 | +.092 | +.004 | +.088 [.015, .156] | .029 | +.104 (.017) |
| smoothing σ = 30 | +.128 | +.010 | +.118 [.004, .223] | .055 | +.126 (.033) |
| zero lag | +.110 | +.006 | +.103 [.012, .189] | .039 | +.123 (.016) |
| lag ±0.5 s | +.110 | +.007 | +.104 [.015, .187] | .033 | +.114 (.021) |
| lag ±1 s | +.107 | +.007 | +.100 [.014, .183] | .036 | +.103 (.030) |

- **Direction holds everywhere.** Δ real is positive with a CI above zero in all 11 variants. Δ MACE is never different from zero (one-sided p .24–.53). G's CI stays above zero in all variants; its p is .013–.056.
- **Stricter matching shrinks the effect by about a third** (G .161 → .104). Real dance's wrong-song score rises from .045 to .100 once the wrong songs really share the tempo, and own > other drops from 15/19 to 13/19 songs. The old matching flattered the human advantage.
- **The lag is not identifiable.** The best lag for real dance sits on the window edge in 86% (±0.3 s), 74% (±0.5 s) and 54% (±1 s) of slices, yet G is the same with zero lag. Report zero-lag r as the main score and drop the lag claim.
- **Floor travel contributes, but isn't the whole effect.** Pelvis-relative joints lower G to .088 (p = .056).
- With 19 songs, an effect of this size sits near p ≈ .03. It needs replication on a second dataset.

## AIST++

### Data (checked 2026-10-27)

- **Motion:** `data/aist/motions/`. Only 411 of the 1408 sequences are still publicly downloadable (GitHub release `motions.zip`): none of the 980 `crossmodal_train` sequences, but all of val/test, all 10 genres, all 60 pieces and all 30 dancers. `aist_convert.py` converts the pickles to `.npz` through a numpy-only unpickler; everything else reads the `.npz` files.
- **Music:** `data/aist/music/` (60 wav files, 48 kHz stereo, 29.5–63 s). This requires the AIST Dance DB application form; the terms allow academic research only and forbid redistribution, so keep it out of git.
- **Tempo** (ISMIR'19 §3.2): every genre uses 80/90/100/110/120/130 BPM (House 110–135). Tempo does not separate genres.
- **Lengths:** every basic dance (sBM) is exactly 16 beats at its piece's tempo (383/383), and every advanced dance (sFM) is exactly 64 beats.
- **Sync:** the refined AIST videos' soundtracks match `wav[0:dur]` at offset 0 (r ≥ 0.999 on 3 videos, including an advanced one whose music runs 10 s past the dance). So motion frame 0 = music time 0, and a dance uses the first 16 or 64 beats of its piece.
- **Joints:** `SMPLSkeleton` FK with root = `smpl_trans / smpl_scaling` (SMPL scale ≈ metres, as in FACT/EDGE). Motion is 60 fps; we take every second frame to get 30 fps.

### Genre × genre matrix, loudness ↔ movement energy (`aist_genre_matrix.py`): no genre link

383 basic dances (ignore list excluded). Clip descriptors: the mean, std and p95−p5 of smoothed RMS dB (music) and of smoothed log10 movement energy (motion). A ridge map from music to motion descriptors is fitted leave-one-genre-out. Cell (A, B) is the distance between the motion predicted from genre-A music and the actual genre-B motion.

- **The diagonal is not closer than chance.** Diagonal 1.87 vs 1.61 under column shuffles (p = 0.99). No row or column has its own genre as the closest match.
- **Loudness does not predict movement energy across genres.** Cross-validated R² is about −1.3 for all three motion descriptors, i.e. worse than predicting the mean. Genre-mean loudness vs genre-mean energy: Spearman ρ = 0.01 (10 genres), ρ = 0.38, p = 0.31 without Ballet Jazz.
- **Ballet Jazz goes the opposite way.** Its music is far quieter than any other genre (−25 dB vs −9 to −14 dB, true of whole pieces, not just intros), yet its dancers move with the highest energy.
- **Loudness varies more within a genre than between genres.** Across the other nine genres, loudness differs more between the 6 pieces of one genre (e.g. Waack −17 to −7 dB) than between genres. Movement energy follows dance style (Street Jazz and Lock lowest, Ballet Jazz, Break and Middle Hip-hop highest).
- **Caveat:** basic dances are 16-beat routines danced unchanged to several pieces, so their energy reflects the routine, not the piece. A genre-level loudness link would have to come through style.

Figures: `results/figures/aist_genre_matrix.png`, `results/figures/aist_genre_scatter.png`.

### Loudness ↔ body energy, real vs MACE (`gen_mace_aist.py`, `aist_loudness.py`, 2026-10-27)

This repeats the FineDance loudness evaluation, including all robustness variants. Units are music pieces. Each piece's music is its first T frames, where T is the shortest real dance on it.
- **Wrong songs:** 3 other-genre pieces at the closest nominal tempo. 198/210 pairs match the tempo exactly; the worst is 4% off.
- **MACE:** 10 samples per piece, generated on the same segment. AIST genres without a FineDance label (Waack, House, Krump) are conditioned as `Urban`.
- **Sets:** basic dances (60 pieces, 16 beats) and advanced dances (10 pieces, 64 beats; first 34 s, inside MACE's 40 s horizon) are analysed separately.

| set | variant | Δ real (p) | Δ MACE (p) | G [95% CI] | G p | real own − MACE own (p) |
|---|---|---|---|---|---|---|
| basic (n = 60) | primary | −.022 (.71) | −.017 (.70) | −.004 [−.108, .101] | .94 | +.051 (.28) |
| basic | random wrong piece, any tempo | +.051 (.08) | −.012 (.65) | +.063 [−.033, .157] | .19 | +.051 (.28) |
| advanced (n = 10) | primary | +.049 (.14) | −.028 (.90) | +.077 [.006, .152] | .075 | +.077 (.006) |
| advanced | zero lag | +.047 (.14) | −.036 (.95) | +.084 [.015, .162] | .054 | +.083 (.006) |

- **Basic dances: no own-song coupling for humans either** (31/60 pieces positive), in every variant. This is expected from the design: each basic routine is danced unchanged to several pieces, so it was never fitted to one piece's loudness. Against random-tempo wrong songs, Δ real rises to +.05 (p = .08), so part of the apparent own-song effect without tempo matching is tempo, not loudness.
- **Advanced dances: the FineDance pattern in direction.** Real dancers score higher with their own piece (7/10 pieces), MACE does not (Δ < 0 in every variant), and real own > MACE own (p = .006). But with 10 pieces, Δ real (p ≈ .14) and G (p ≈ .05–.3 across variants) are not significant.
- **Short clips inflate lag-max r.** At ±1 s lag, basic r rises from .13 to .33 for own and wrong music alike. Δ is unaffected, but raw S values from 7–12 s clips shouldn't be compared with FineDance's.
- **MACE motion scale:** mean joint speed 0.6–0.7 m/s vs 0.9–1.1 m/s for real dance; root travel std 0.10 m vs 0.22–0.42 m.

## FineDance public release (2026-10-27)

### Data and checks (`finedance_common.py`)

- **Source:** the authors' public release (Google Drive link in the FineDance README; 7.7 of the 14.6 h), in `data/finedance/finedance/`. It holds 203 songs with motion, wav and label, including all 19 MACE test songs. Motion is `(T, 315)` at 30 fps: root position plus 52 joints × 6D rotation (FineDance `data/code/pre_motion.py`). The paper (ICCV'23, Fig. 2) says dancers aligned music and motion *by hand* in Blender.
- **Mapping to MACE's 24-joint SMPL:** release joints 0–21 plus 25 and 40 (the first joint of each hand). On the test songs this gives the same root position and the same rotations to 1e-7.
- **The shipped test files have glitches.** In 5 of 19 test songs (012, 036, 037, 109, 143), `Expert-Motion/data/test/motions*` contain single-frame root-orientation jumps, 9–50 frames per song, that move joints at up to 28.6 m/s. The release is smooth there; the other 13 songs agree to within 1.2 cm. `robustness_loud.py --gt release` rebuilds the real-dance slices (1020 frames every 510) from the release.
- **Duplicate music.** 20 pairs of songs share one recording: duets (the `shuangren` pairs, 104–147), 070/071, and 096/165 (offset by 0.03 s). In 6 test songs (109, 130, 132, 137, 143, 144), the twin (108, 131, 133, 136, 142, 145) is in MACE's training set (`processed_train_data.pkl`, 183 songs). So MACE trained on that music, danced by someone else.

### Test-set loudness result with clean real-dance motion (`robustness_loud.py --gt release`)

Slightly stronger than with the test files. Primary: Δ real +.117, Δ MACE +.006, G +.111 [.027, .191], p = .019; real own − MACE own +.129, p = .009. G has p < .05 in all 11 variants (it was 9/11 before).

The music leak does not help MACE: Δ MACE is +.009 on the 6 leaked songs and +.005 on the 13 others (p = .90). G is larger on the clean songs (+.138 vs +.050).

### Replication on non-test songs, real dance only (`finedance_release_loud.py`)

The pool rules were set before the results were seen:
- release songs not in the test set and not sharing music with it;
- motion and music lengths within 1 s (drops 041, 043, 052, 053, 059);
- at least one 34 s slice;
- not silent (drops 169, whose wav is 92% digital silence).

That leaves 165 songs in **151 music pieces** (duets merged) and 1143 slices. Wrong songs are tempo-first matches from the same pool (median tempo gap 0.5%, max 3%), never a song's own duplicate. MACE is left out because it was trained on these songs.

| variant | real own | real other | Δ real [95% CI] | pieces Δ > 0 |
|---|---|---|---|---|
| **primary** | .185 | .070 | **+.114 [.084, .145]** | 113/151 |
| random wrong piece (any tempo) | .185 | .070 | +.115 [.085, .146] | 113/151 |
| RMS dB / A-weighted dB | .187 / .178 | .073 / .073 | +.114 / +.105 | 110 / 114 |
| mean speed / pelvis-relative | .221 / .185 | .075 / .070 | +.147 / +.115 | 119 / 114 |
| σ = 8 / σ = 30 | .166 / .212 | .070 / .074 | +.096 / +.138 | 112 / 103 |
| zero lag / ±0.5 s / ±1 s | .156 / .202 / .238 | .040 / .090 / .131 | +.116 / +.112 / +.108 | 111 / 115 / 115 |

One-sided p < .0001 (permutation and Wilcoxon) in every variant.

- **The human effect replicates at the same size as on the test set** (+.114 vs +.117), now on 151 independent pieces.
- **By genre:** Korean, Popping, Jazz and Hip-hop are strongest (Δ +.15 to +.18, 75–94% of pieces positive). ShenYun (+.04, 26 pieces) and Breaking (−.04, 10 pieces) are weakest. Loudness coupling is a strong cue in pop and street styles, and a weak one in Chinese classical and breaking.
