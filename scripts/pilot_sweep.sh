#!/usr/bin/env bash
# anGPT vs GPT+ pilot, run sequentially on one GPU.
#
#   Stage 1  LR sweep: 3 LRs per arm, 100M tokens (1526 steps), seed 1
#            bash scripts/pilot_sweep.sh lr
#   Stage 2  confirm: best LR per arm, 300M tokens (4578 steps), seeds 1 and 2
#            bash scripts/pilot_sweep.sh confirm <gptplus_lr> <angpt_lr>
#
# Decide from final/loss (after full LR decay), never from mid-run curves.
# If the best LR is at the edge of a grid, extend the grid before stage 2.
set -euo pipefail
cd "$(dirname "$0")/.."

TRAIN="env PYTHONPATH=.:${PYTHONPATH:-} python ramanujan/training/train_candide18c.py --wandb"
GPT_LRS=(3e-4 6e-4 1.2e-3)
ANGPT_LRS=(1e-3 2e-3 4e-3)

case "${1:-}" in
  lr)
    for lr in "${GPT_LRS[@]}"; do
      $TRAIN --config configs/pilot/gptplus.yaml --lr "$lr" --seed 1 \
             --max-steps 1526 --warmup-steps 153 \
             --run-name "lr-gptplus-${lr}-s1" --group pilot-lr-sweep
    done
    for lr in "${ANGPT_LRS[@]}"; do
      $TRAIN --config configs/pilot/angpt.yaml --lr "$lr" --seed 1 \
             --max-steps 1526 --warmup-steps 0 \
             --run-name "lr-angpt-${lr}-s1" --group pilot-lr-sweep
    done
    ;;
  confirm)
    gpt_lr="${2:?gptplus LR required}"; angpt_lr="${3:?angpt LR required}"
    for seed in 1 2; do
      $TRAIN --config configs/pilot/gptplus.yaml --lr "$gpt_lr" --seed "$seed" \
             --run-name "gptplus-${gpt_lr}-s${seed}" --group pilot-confirm
      $TRAIN --config configs/pilot/angpt.yaml --lr "$angpt_lr" --seed "$seed" \
             --run-name "angpt-${angpt_lr}-s${seed}" --group pilot-confirm
    done
    ;;
  *)
    echo "usage: $0 lr | confirm <gptplus_lr> <angpt_lr>"; exit 1 ;;
esac
