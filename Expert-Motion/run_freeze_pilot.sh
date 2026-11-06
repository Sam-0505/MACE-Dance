#!/bin/bash
# Root rotation frozen in the beat-locking guidance (2026-11-06), with the foot-contact term (10 steps):
# lambda 0.003 (10 seeds: also the video / final candidate) and 0.005 (3 seeds), FineDance; against mask + guidance +
# foot without the freeze (10 seeds; paired seeds). Scores: authors' metrics, coupling, Lock / BAS, FSR, VBCS / ABHS,
# rotation jitter; videos of the two earlier clips.
cd "$(dirname "$0")"
E=../Evaluation-Musicality
P=.venv/bin/python
L=logs/freeze_pilot; mkdir -p $L
BASE=mask2s_guide10_s0.003_foot10_10seeds
for S in 0.003 0.005; do
  N=3; [ $S = 0.003 ] && N=10
  R=mask2s_guide10_s${S}_foot10_frz
  $P $E/gen_mace_test.py --seeds $N --local_attn 2 --lock_guide $S --lock_foot 10 --lock_freeze_root --out_dir $E/data/mace_test_$R > $L/gen_$R.log 2>&1 || echo "gen $R failed" >> $L/status.log
  $P $E/authors_metrics.py --mace_dir $E/data/mace_test_$R > $L/authors_$R.log 2>&1
  $P $E/finetune_eval.py --label $R --mace_dir $E/data/mace_test_$R > $L/ftEval_$R.log 2>&1
  echo "$R done" >> $L/status.log
done
$P $E/beat_suite.py --dataset finedance --tag _freeze_pilot --fd_models foot=mace_test_$BASE \
  foot_frz3=mace_test_mask2s_guide10_s0.003_foot10_frz foot_frz5=mace_test_mask2s_guide10_s0.005_foot10_frz > $L/beat_suite.log 2>&1
$P $E/fsr_table.py --tag _freeze_pilot --fd foot=mace_test_$BASE foot_frz3=mace_test_mask2s_guide10_s0.003_foot10_frz \
  foot_frz5=mace_test_mask2s_guide10_s0.005_foot10_frz > $L/fsr.log 2>&1
$P $E/tmd_bench_table.py --tag _freeze_pilot --fd foot=mace_test_$BASE foot_frz3=mace_test_mask2s_guide10_s0.003_foot10_frz \
  foot_frz5=mace_test_mask2s_guide10_s0.005_foot10_frz > $L/tmd.log 2>&1
$P $E/rotation_jitter.py --tag _freeze_pilot --seeds 3 --fd released=mace_test mask_guide=mace_test_mask2s_guide10_s0.003 \
  foot=mace_test_$BASE foot_frz3=mace_test_mask2s_guide10_s0.003_foot10_frz foot_frz5=mace_test_mask2s_guide10_s0.005_foot10_frz > $L/jitter.log 2>&1
FF=/home/sam0505/Documents/Code/MACE-Dance/Motion-Projector/.venv/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2
$P $E/render_fix_compare.py --ffmpeg $FF --oneshot_label 'MACE released' --extra $E/data/mace_test_$BASE 'Mask + guide + foot' \
  --fixed_dir $E/data/mace_test_mask2s_guide10_s0.003_foot10_frz --fixed_label '+ root rotation frozen' \
  --suffix real_released_foot_frozen > $L/videos.log 2>&1
echo "pilot finished" >> $L/status.log
