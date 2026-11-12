# Leave-one-out ablation, seed 1 (2026-11-10): how the table was produced

Generation: `Expert-Motion/run_ablation.sh` (variants nomask / nocritic / nofoot / nofrz; the full flags were checked to reproduce the
stored gain10 run exactly, max |diff| 0.0 m). Scoring, all from `Expert-Motion` with `.venv/bin/python`, rows = released
(`mace_test`), `mace_test_local2s`, `mace_test_mask2s_criticv5_s0.003_foot10_frz` (Full - gain), the four `mace_test_abl_*`, and
`mace_test_mask2s_criticv5_s0.003_foot10_frz_gain10` (Full):

    authors_metrics.py --mace_dir ../Evaluation-Musicality/data/<dir> --seeds 1 --out ../Evaluation-Musicality/results/authors_metrics_ablation.json
    fsr_table.py --fd released=mace_test mask=mace_test_local2s nogain=<dir> full=<dir> nomask=... nocritic=... nofoot=... nofrz=... --mot --seeds 1 --tag _abl1
    lock_models.py --dataset finedance --seeds 1 --models "MACE released,MACE mask2s,Ours critic v5,Ours critic v5 + gain,Abl - mask,Abl - critic,Abl - foot,Abl - frozen root" --tag _abl1
    loud_models.py   (1 seed; its output loud_models_1seed.json was renamed to loud_models_finedance_seed1_abl.json and the older file restored)
    ablation_table.py -> results/ablation_table.{json,md}

`authors_metrics.py` gained a `--seeds N` option for this. `lock_models.py` / `loud_models.py` gained the four `Abl - ...` model labels.
