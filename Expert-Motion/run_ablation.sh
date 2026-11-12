#!/bin/bash
# Leave-one-out ablation of the headline model (2026-11-10): full = mask 2 s + critic v5 guidance (0.003) + foot term (10) +
# frozen root rotation + gain (0.49), FineDance 84 slices, 10 seeds (same seeds/noise as the stored runs -> paired).
# Settings check: the CLI with the full flags reproduced the stored gain10 seed 0 exactly (max |diff| 0.0 m).
#   - mask:   full attention (no --local_attn)
#   - critic: no critic push: --lock_guide 1e-9 (the foot term only runs inside the guidance block, so the guidance is
#             kept with a vanishing scale; gain and frozen root only act on the push, so they have no effect here)
#   - foot:   --lock_foot 0
#   - frz:    no --lock_freeze_root
# NOTE 2026-11-10: the user then asked for ONE seed. The script was stopped while 'nocritic' was generating; nomask, nofoot and
# nofrz had finished with 10 seeds (only seed 0 is used in the table), nocritic was regenerated with --seeds 1.
# Logs: logs/ablation/.   Outputs: ../Evaluation-Musicality/data/mace_test_abl_{nomask,nocritic,nofoot,nofrz}
cd "$(dirname "$0")"
E=../Evaluation-Musicality
P=.venv/bin/python
L=logs/ablation; mkdir -p $L
CK=$E/data/critic/critic_v5.pt
export WANDB_MODE=offline
log() { echo "$(date +%T) $*" >> $L/status.log; }
gen() {  # name, extra flags...
  n=$1; shift
  $P $E/gen_mace_test.py --seeds 10 --critic_guide --critic_ckpt $CK "$@" --out_dir $E/data/mace_test_abl_$n > $L/gen_$n.log 2>&1 \
    && log "generated $n" || log "gen $n FAILED"
}
log "start"
gen nomask   --lock_guide 0.003 --lock_foot 10 --lock_freeze_root --gain_ref 0.49
gen nofoot   --local_attn 2 --lock_guide 0.003 --lock_freeze_root --gain_ref 0.49
gen nofrz    --local_attn 2 --lock_guide 0.003 --lock_foot 10 --gain_ref 0.49
gen nocritic --local_attn 2 --lock_guide 1e-9 --lock_foot 10 --lock_freeze_root --gain_ref 0.49
log "ALL GENERATED"
