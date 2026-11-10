#!/bin/bash
# Critic v5 (2026-11-08): low-capacity beat-synchronous critic (critic4.py --lite: 4 body-part speeds, non-negative linear
# events, learned phase template, offset-free), ONE critic (user: start simple; copies / bigger nets / data later if it works a
# little). Pre-declared: sample only if it passes critic_gate4.py. Mask +-2 s + critic guidance + foot term + frozen root
# rotation (lite inputs are root-relative). 1 seed: FineDance, placebo, Motorica. Success = PLV delta (world and root-relative)
# above released with p < 0.05, placebo at chance, quality near final. The critic's own score on the guided dances is logged
# as a gaming check (far above real dance with flat PLV = gamed). Logs: logs/critic_v5/.
cd "$(dirname "$0")"
E=../Evaluation-Musicality
P=.venv/bin/python
L=logs/critic_v5; mkdir -p $L
CK=$E/data/critic/critic_v5.pt
FINAL=mace_test_mask2s_guide10_s0.003_foot10_frz
log() { echo "$(date +%T) $*" >> $L/status.log; }
log "start"

$P $E/critic4.py --lite --lr 1e-2 --seed 0 --out $CK > $L/train.log 2>&1 || log "train failed"
$P $E/critic_gate4.py --critic $CK --tag _v5 > $L/gate.log 2>&1 || log "gate failed"
if ! grep -q "GATE PASSED" $L/gate.log; then log "GATE FAILED (see $L/gate.log); sampling not run"; exit 0; fi
log "gate passed"
FLAGS="--local_attn 2 --lock_foot 10 --lock_freeze_root --critic_guide --critic_ckpt $CK"

R=mask2s_criticv5_s0.003_foot10_frz
$P $E/gen_mace_test.py --seeds 1 --lock_guide 0.003 $FLAGS --out_dir $E/data/mace_test_$R > $L/gen_$R.log 2>&1 || log "gen $R failed"
log "generated $R"
$P $E/authors_metrics.py --mace_dir $E/data/mace_test_$R --out $E/results/authors_metrics_criticv5.json > $L/authors_$R.log 2>&1
$P $E/beat_suite.py --dataset finedance --tag _criticv5 --tempo_null --fd_models final=$FINAL critic_v5=mace_test_$R > $L/beat_suite_$R.log 2>&1
$P $E/beat_suite.py --dataset finedance --tag _criticv5_rootrel --root_relative --tempo_null --fd_models final=$FINAL critic_v5=mace_test_$R > $L/beat_suite_rootrel_$R.log 2>&1
$P $E/fsr_table.py --tag _criticv5_fd --fd final=$FINAL critic_v5=mace_test_$R > $L/fsr_$R.log 2>&1
$P $E/finetune_eval.py --label $R --mace_dir $E/data/mace_test_$R > $L/ftEval_$R.log 2>&1
$P $E/critic_gate4.py --critic $CK --tag _v5_guided --sets final=$FINAL critic_v5=mace_test_$R > $L/critic_on_guided.log 2>&1
log "scored $R"

PL=${R}_placebo
$P $E/gen_mace_test.py --seeds 1 --lock_guide 0.003 --lock_guide_shift 10 $FLAGS --out_dir $E/data/mace_test_$PL > $L/gen_$PL.log 2>&1 || log "gen $PL failed"
$P $E/beat_suite.py --dataset finedance --tag _criticv5_placebo --tempo_null --fd_models critic_v5=mace_test_$R placebo=mace_test_$PL > $L/beat_suite_$PL.log 2>&1
log "scored placebo"

M=mace_local2s_criticv5_s0.003_foot10_frz
$P $E/gen_mace_motorica.py --seeds 1 --local_attn 2 --lock_guide 0.003 --lock_foot 10 --lock_freeze_root --critic_guide \
  --critic_ckpt $CK --out_name $M > $L/gen_mot.log 2>&1 || log "gen motorica failed"
FM=mace_local2s_guide_s0.003_foot10_frz
$P $E/beat_suite.py --dataset motorica --tag _criticv5 --mot_models final=$FM critic_v5=$M > $L/beat_suite_mot.log 2>&1
$P $E/beat_suite.py --dataset motorica --tag _criticv5_rootrel --root_relative --mot_models final=$FM critic_v5=$M > $L/beat_suite_mot_rootrel.log 2>&1
$P $E/fsr_table.py --tag _criticv5_mot --mot critic_v5=$M > $L/fsr_mot.log 2>&1
log "ALL DONE"
