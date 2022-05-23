#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
FOLD=$3
PERCENT=$4
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
torchrun --nproc_per_node=$GPUS $(dirname "$0")/train.py \
    $CONFIG --cfg-options fold=${FOLD} percent=${PERCENT} ${@:5}
