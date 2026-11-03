#!/bin/bash
# A + B fine-tune, then the standard evaluation; detached-safe (setsid nohup).
cd "$(dirname "$0")"
./order_ab.sh && ./eval_long.sh long_ab_l3e-3 > logs/eval_ab.log 2>&1
echo "run_ab_all finished, exit $?" >> logs/run_ab_all.log
