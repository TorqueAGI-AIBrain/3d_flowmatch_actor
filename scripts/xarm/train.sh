#!/bin/bash
# train_real_arm.sh: Launch training using YAML config.
# Usage: bash scripts/real_arm/train_real_arm.sh [configs/training.yaml] [--extra_cli_overrides]

CONFIG=${1:-configs/training.yaml}
shift 2>/dev/null  # remaining args passed as CLI overrides

ngpus=1

torchrun --nproc_per_node $ngpus --master_port $RANDOM \
    main.py \
    --config $CONFIG \
    "$@"
