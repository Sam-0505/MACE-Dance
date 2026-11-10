#!/bin/bash
# Critic v5 full run (2026-11-08): mask +-2 s + critic v5 guidance + foot term + frozen root rotation, FineDance 10 seeds,
# placebo 5 seeds, Motorica 3 seeds (as the final model), then beat locking (world and root-relative joints), foot sliding and
# the critic's own score, compared with released / mask / final (formula) / EDGE / Lodge / Bailando / Lodge+our guidance.
# Overwrites the 1-seed run's outputs (seed 0 is identical). Logs: logs/critic_v5_full/.
cd "$(dirname "$0")"
E=../Evaluation-Musicality
P=.venv/bin/python
L=logs/critic_v5_full; mkdir -p $L
CK=$E/data/critic/critic_v5.pt
FINAL=mace_test_mask2s_guide10_s0.003_foot10_frz
log() { echo "$(date +%T) $*" >> $L/status.log; }
FLAGS="--local_attn 2 --lock_foot 10 --lock_freeze_root --critic_guide --critic_ckpt $CK"
R=mask2s_criticv5_s0.003_foot10_frz
M=mace_local2s_criticv5_s0.003_foot10_frz
log "start"

$P $E/gen_mace_test.py --seeds 10 --lock_guide 0.003 $FLAGS --out_dir $E/data/mace_test_$R > $L/gen_fd.log 2>&1 || log "gen fd failed"
log "generated FineDance"
$P $E/gen_mace_test.py --seeds 5 --lock_guide 0.003 --lock_guide_shift 10 $FLAGS --out_dir $E/data/mace_test_${R}_placebo > $L/gen_placebo.log 2>&1 || log "gen placebo failed"
log "generated placebo"
$P $E/gen_mace_motorica.py --seeds 3 --local_attn 2 --lock_guide 0.003 --lock_foot 10 --lock_freeze_root --critic_guide \
  --critic_ckpt $CK --out_name $M > $L/gen_mot.log 2>&1 || log "gen motorica failed"
log "generated Motorica"

FD="mask_2s=mace_test_local2s final=$FINAL critic_v5=mace_test_$R edge=mace_test_edge lodge=mace_test_lodge bailando=mace_test_bailando lodge_guided=mace_test_lodge_guide_frz_foot"
MOT="mask_2s=mace_local2s final=mace_local2s_guide_s0.003_foot10_frz critic_v5=$M edge=edge lodge=lodge bailando=bailando"
$P $E/beat_suite.py --dataset finedance --tag _cmp_v5 --tempo_null --fd_models $FD > $L/beat_fd.log 2>&1
$P $E/beat_suite.py --dataset finedance --tag _cmp_v5_rootrel --root_relative --tempo_null --fd_models $FD > $L/beat_fd_rootrel.log 2>&1
$P $E/beat_suite.py --dataset finedance --tag _cmp_v5_placebo --tempo_null --fd_models critic_v5=mace_test_$R placebo=mace_test_${R}_placebo > $L/beat_placebo.log 2>&1
$P $E/beat_suite.py --dataset motorica --tag _cmp_v5 --mot_models $MOT > $L/beat_mot.log 2>&1
$P $E/beat_suite.py --dataset motorica --tag _cmp_v5_rootrel --root_relative --mot_models $MOT > $L/beat_mot_rootrel.log 2>&1
log "beat scored"
$P $E/authors_metrics.py --mace_dir $E/data/mace_test_$R --out $E/results/authors_metrics_criticv5.json > $L/authors.log 2>&1
$P $E/fsr_table.py --tag _cmp_v5_fd --fd $FD > $L/fsr_fd.log 2>&1
$P $E/fsr_table.py --tag _cmp_v5_mot --mot $MOT > $L/fsr_mot.log 2>&1
$P $E/critic_gate4.py --critic $CK --tag _v5_guided10 --sets final=$FINAL critic_v5=mace_test_$R > $L/critic_on_guided.log 2>&1
log "ALL DONE"
