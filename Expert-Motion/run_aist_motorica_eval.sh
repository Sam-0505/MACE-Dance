#!/bin/bash
# Motorica (out of domain) scores of EDGE, Bailando and the AIST++-trained MACE (critic + gain, 1 seed; gen_aist_model_motorica.py),
# on the 289 non-overlapping 34 s slices. FID_k / DIV_k / FSR / BAS / Lock / Loud as in run_frz_motorica.sh and motorica_main_table.py.
cd "$(dirname "$0")"
E=../Evaluation-Musicality
P=.venv/bin/python
L=logs/aist/motorica; mkdir -p $L
export WANDB_MODE=offline
$P $E/motorica_compare.py --tag _aist --models edge=edge bailando=bailando ours_aist=ours_aist > $L/compare.log 2>&1 || echo "compare failed" >> $L/status.log
$P $E/beat_suite.py --dataset motorica --tag _aist --mot_models edge=edge bailando=bailando ours_aist=ours_aist > $L/beat_suite.log 2>&1 || echo "beat_suite failed" >> $L/status.log
$P $E/fsr_table.py --tag _aist_mot --mot edge=edge bailando=bailando ours_aist=ours_aist > $L/fsr.log 2>&1 || echo "fsr failed" >> $L/status.log
$P $E/lock_models.py --dataset motorica --seeds 1 --models "EDGE,Bailando,Ours AIST++" --tag _aist > $L/lock.log 2>&1 || echo "lock failed" >> $L/status.log
$P $E/loud_models_motorica.py --seeds 1 --only "EDGE,Bailando,Ours AIST++" > $L/loud.log 2>&1 || echo "loud failed" >> $L/status.log
echo "all done" >> $L/status.log
