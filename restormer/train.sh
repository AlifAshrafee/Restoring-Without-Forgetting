#!/usr/bin/env bash
# Multi-GPU training script for Restormer
# Usage: ./train.sh <config.yml>

CONFIG=$1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

torchrun --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt $CONFIG --launcher pytorch
