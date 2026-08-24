#!/usr/bin/env bash
# Single-GPU training script for Restormer
# Usage: ./train_single_gpu.sh <config.yml>

CONFIG=$1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

python basicsr/train.py -opt $CONFIG
