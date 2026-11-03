#!/bin/bash
# Long-clip fine-tunes then their evaluation; detached so it survives the Claude session (2026-11-03).
cd "$(dirname "$0")"
./order_long.sh && ./eval_long.sh > logs/eval_long.log 2>&1
echo "run_long_all finished, exit $?" >> logs/run_long_all.log
