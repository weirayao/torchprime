#!/bin/bash

# Check if required arguments are provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <checkpoint_load_dir> <checkpoint_load_step>"
    echo "       checkpoint_load_step can be a single step or comma-separated list of steps"
    echo "       Each checkpoint will be consolidated separately into its own directory"
    echo "Examples:"
    echo "  Single checkpoint: $0 gs://sfr-text-diffusion-model-research/checkpoints/flex_processed_v1_qw1_7b_512_split_datafix 32000"
    echo "  Multiple checkpoints: $0 gs://sfr-text-diffusion-model-research/checkpoints/flex_processed_v1_qw1_7b_512_split_datafix \"16000,24000,32000\""
    exit 1
fi

# Please only run this script on single-host TPU (v4-8/v5-8)

# Assign arguments to variables
CHECKPOINT_DIR=$1
RESUME_FROM_CHECKPOINT=$2

XLA_IR_DEBUG=1 XLA_HLO_DEBUG=1 python torchprime/torch_xla_models/ckpt_consolidation.py \
    model=flex-qwen-1b \
    checkpoint_load_dir=${CHECKPOINT_DIR} \
    checkpoint_load_step=${RESUME_FROM_CHECKPOINT} \
    ici_mesh.fsdp=4 \
    ici_mesh.tensor=1 \
    ici_mesh.data=1 \
    ici_mesh.expert=1 \
    model/remat=qwen-scan
# fsdp * tensor * data * expert == num_devices
# global_batch_size mod num_devices == 0
