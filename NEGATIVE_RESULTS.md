# Negative results and removed code

Cleanup of 2026-11-10. Everything below was tried, did not make it into the method, and was **removed from the working tree**.
The code is intact at git tag **`pre-cleanup-2026-11-10`** (`git checkout pre-cleanup-2026-11-10 -- <path>` restores a file).
All result JSONs in `Evaluation-Musicality/results/` were kept.

**Shipped method (training-free):** released checkpoint + ±2 s cross-attention mask + critic v5 guidance + foot term + frozen root
rotation + motion-proportional gain (`gen_mace_test.py --local_attn 2 --lock_guide 0.003 --lock_foot 10 --lock_freeze_root
--critic_guide --gain_ref 0.49`). The locking-formula guidance (`model/beatlock.py: windowed_locking`) is kept as a reference variant.

## Training-time losses (fine-tunes from the released checkpoint)

| Approach | What it was | Outcome | Results | Removed code |
|---|---|---|---|---|
| Slow-coupling loss (`--lambda_dyn`) | 1 − r(music, body energy) of predicted x0; treble/perc/loud, hinge and low-pass variants | The ±2 s mask closes the slow-coupling gap without training, so the loss had no target left | `finetune_sweep.json` | `model/musicality.py` loss terms, `order_long.sh`, `order_ab.sh` |
| Beat-locking loss (`--lambda_lock`) | hinge max(0, D_real − D_pred), D = chance-corrected offset-free locking of speed peaks to the beats; pilot 1 and pilot 2 (loss only at high t, low-passed joints) | x0 locking rose only at low noise (sharpening the input); sampled dances showed no locking (PLV delta ≤ +0.008 vs real +0.12) and quality dropped | `beat_suite_finedance_lock*.json`, `beatlock_validate*.json` | `locking_terms` in `model/beatlock.py`, `order_lock*.sh`, `run_lock*_all.sh` |
| Beat-phase input (`--beat_phase_input`) | zero-initialised cos/sin beat-phase projection added to the music conditioning (arm B) | no locking in samples | same as above | `phase_projection` in `model/model.py` |
| Contrastive locking (`--lambda_con`) | InfoNCE of own-beat vs other-song locking score of predicted x0, LoRA r16 + beat-phase block | train-time own−other gap rose, sampled PLV delta −0.002 (real +0.122) | `beat_suite_finedance_con_probe.json`, `authors_metrics_con_probe.json` | `contrastive_locking`, `order_con.sh` |
| Long-clip fine-tune, full and LoRA | 34 s training clips (`--long_clips`, `--lora_rank`), λ = 0 controls | the λ = 0 fine-tune already shows jitter / foot-skate damage; the mask fixes long-clip coupling at inference | `length_curve_long*.json`, `length_curve_lora*.json`, `attention_check_lora_l0.json` | `model/lora.py`, `dataset/long_clips.py`, `order_long.sh`, `order_lora.sh`, `run_long_all.sh`, `run_lora_all.sh`, `run_window_diag.sh`, `eval_long.sh` |

`build_long_clips.py` and `precompute_music_signals.py` were kept: `full_song_curve.py` imports `baseline_features` from the first.

## Critic guidance, earlier versions

Critics v1–v4 had their code removed on 2026-11-08, and on 2026-11-10 their checkpoints, training logs and generated samples were
deleted from disk too (gitignored, not recoverable; only the result JSONs below survive). Only the low-capacity v5 critic remains
(`critic4.py --lite`). v1–v3 failed their validity gates (held-out wrong-time accuracy 0.55–0.57); v4 passed its gate but guidance was gamed
by root sliding (root-detached variant: quality recovered, PLV gain gone). Results: `critic_validate*.json`, `critic_timing_*.json`,
`critic_gate4*.json`, `beat_suite_finedance_critic*.json`, `beat_suite_finedance_criticv4*.json`, `authors_metrics_critic*.json`.

## Beat-bead / speed-shading visualisations

Beat beads on limb trails, per-beat photo strips, phase grids, relative-speed and absolute-speed body shading, `shade_curve` options.
The beat-rate-wave shading and phase clock/graph (`render_pink_beads.py`) replaced them: zero-offset speed-at-beat does not show offset-free
locking. Removed: `render_beads*.py`, `render_beat_photos*.py`, `rank_beat_*.py`, `rank_hits_on_beat.py`, `rank_relspeed_clips.py`,
`find_real_stretch.py`, and the strip / photo-grid / glow-by-speed modes of `render_pink_beads.py`. Looped-song and drum-track probes
(`gen_loop_test.py`, `gen_drum_test.py`; `results/loop_test.json`, `results/drum_test.json`) showed neither model repeats steps when the music repeats.
Old videos were moved to `Evaluation-Musicality/data/videos/_old_beads/` (gitignored).
