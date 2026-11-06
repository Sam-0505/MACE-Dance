#!/bin/bash
# Lock with the tempo-matched null for the earlier versions quoted in slide notes, 2026-11-06
P=.venv/bin/python; E=../Evaluation-Musicality
$P $E/beat_suite.py --dataset finedance --tempo_null --tag _tempo_null_versions \
  --fd_models mask_guide=mace_test_mask2s_guide10_s0.003 foot=mace_test_mask2s_guide10_s0.003_foot10_10seeds > $E/results/tempo_null_versions_fd.log 2>&1
$P $E/beat_suite.py --dataset motorica --tempo_null --tag _tempo_null_versions \
  --mot_models mask_guide=mace_local2s_guide_s0.003 foot=mace_local2s_guide_s0.003_foot10 > $E/results/tempo_null_versions_mot.log 2>&1
echo done
