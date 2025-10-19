# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#!/bin/bash

NUM_GPUS=4
EXPERIMENT_NAME=baseline
export OMP_NUM_THREADS=8  # adjust according to your system

torchrun --nnodes=1 --nproc_per_node=$NUM_GPUS scripts/train.py configs/default.yaml --exp_dir=experiments/$EXPERIMENT_NAME