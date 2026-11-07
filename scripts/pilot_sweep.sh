#!/usr/bin/env bash
# anGPT vs GPT+ pilot, run sequentially on one GPU.
#
#   Stage 1  LR sweep: 3 LRs per arm, 100M tokens (1526 steps), seed 1
#            bash scripts/pilot_sweep.sh lr
#            (v2, 2026-11-07: global data order + fixed val sample; grid moved up
#             one step after v1 put both optima at the top edge)
#   Stage 2  confirm: best LR per arm, 300M tokens (4578 steps), seeds 1 and 2
#            bash scripts/pilot_sweep.sh confirm <gptplus_lr> <angpt_lr>
#
# Decide from final/loss (after full LR decay), never from mid-run curves.
# If the best LR is at the edge of a grid, extend the grid before stage 2.
set -euo pipefail
cd "$(dirname "$0")/.."

TRAIN="env PYTHONPATH=.:${PYTHONPATH:-} python ramanujan/training/train_candide18c.py --wandb"
GPT_LRS=(6e-4 1.2e-3 2.4e-3)
ANGPT_LRS=(2e-3 4e-3 8e-3)

case "${1:-}" in
  lr)
    for lr in "${GPT_LRS[@]}"; do
      $TRAIN --config configs/pilot/gptplus.yaml --lr "$lr" --seed 1 \
             --max-steps 1526 --warmup-steps 153 \
             --run-name "lr2-gptplus-${lr}-s1" --group pilot-lr-sweep-v2
    done
    for lr in "${ANGPT_LRS[@]}"; do
      $TRAIN --config configs/pilot/angpt.yaml --lr "$lr" --seed 1 \
             --max-steps 1526 --warmup-steps 0 \
             --run-name "lr2-angpt-${lr}-s1" --group pilot-lr-sweep-v2
    done
    ;;
  confirm)
    gpt_lr="${2:?gptplus LR required}"; angpt_lr="${3:?angpt LR required}"
    for seed in 1 2; do
      $TRAIN --config configs/pilot/gptplus.yaml --lr "$gpt_lr" --seed "$seed" \
             --run-name "gptplus-${gpt_lr}-s${seed}" --group pilot-confirm-v2
      $TRAIN --config configs/pilot/angpt.yaml --lr "$angpt_lr" --seed "$seed" \
             --run-name "angpt-${angpt_lr}-s${seed}" --group pilot-confirm-v2
    done
    ;;
  *)
    echo "usage: $0 lr | confirm <gptplus_lr> <angpt_lr>"; exit 1 ;;
esac
