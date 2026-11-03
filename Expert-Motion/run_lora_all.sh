#!/bin/bash
# Waits for run_ab_all.sh (one GPU), then the LoRA fine-tunes and their evaluation; detached-safe (setsid nohup).
cd "$(dirname "$0")"
until grep -q "run_ab_all finished" logs/run_ab_all.log 2>/dev/null; do sleep 60; done
./order_lora.sh && ./eval_long.sh lora_l0 lora_ab_l3e-3 > logs/eval_lora.log 2>&1
echo "run_lora_all finished, exit $?" >> logs/run_lora_all.log
