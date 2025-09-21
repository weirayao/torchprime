#!/bin/bash

TPU_VM_NAME="sfr-haolin-chen-v5p-8"
TPU_ZONE="us-central1-a"
BRANCH="haolin/pretrain_qwen25_coder_hyperparam_tuning"
RECIPE="recipes/ckpt_consolidation.sh"

# Define checkpoint directories with their corresponding model configs and resume checkpoints
GCS_PREFIX="gs://sfr-text-diffusion-model-research/checkpoints/"

# Associative arrays mapping checkpoint directories to their model configs and resume checkpoints
declare -A MODEL_CONFIG
declare -A CHECKPOINT_CONFIG

# Configure each checkpoint directory with its model and resume checkpoints
# Format: MODEL_CONFIG["checkpoint_dir"]="model_name"
#         CHECKPOINT_CONFIG["checkpoint_dir"]="[step1 step2 step3]"
MODEL_CONFIG["pretrain_qwen25_coder_tpu_128_context_8192_segment_attn_small_batch_lr_3e-4"]="flex-qwen2-1b"
CHECKPOINT_CONFIG["pretrain_qwen25_coder_tpu_128_context_8192_segment_attn_small_batch_lr_3e-4"]="[70000,80000,90000,100000]"

MODEL_CONFIG["pretrain_qwen3_tpu_64_context_8192_full_attn_small_batch_lr_3e-4"]="flex-qwen-1b"
CHECKPOINT_CONFIG["pretrain_qwen3_tpu_64_context_8192_full_attn_small_batch_lr_3e-4"]="[70000,80000,90000,100000]"

# Add more checkpoint directories here:
# MODEL_CONFIG["another_checkpoint_dir"]="different-model-config"
# CHECKPOINT_CONFIG["another_checkpoint_dir"]="[1000,5000,10000]"

for checkpoint_dir in "${!CHECKPOINT_CONFIG[@]}"; do
    model="${MODEL_CONFIG[$checkpoint_dir]}"
    resume_checkpoints="${CHECKPOINT_CONFIG[$checkpoint_dir]}"
    checkpoint="${GCS_PREFIX}${checkpoint_dir}"
    echo "=================================================="
    echo "Running checkpoint consolidation:"
    echo "  Checkpoint dir: $checkpoint"
    echo "  Model config: $model"
    echo "  Resume checkpoint: $resume_checkpoints"
    echo "=================================================="
    # Run the gcloud command and wait for it to complete
    gcloud alpha compute tpus tpu-vm ssh $TPU_VM_NAME \
        --zone=$TPU_ZONE \
        --project=salesforce-research-internal \
        --tunnel-through-iap \
        --worker=all \
        --command='
        cd torchprime; \
        git fetch; \
        git checkout '"$BRANCH"'; \
        git pull; \
        source venv/bin/activate; \
        bash '"$RECIPE"' '"$model"' '"$checkpoint"' "'"$resume_checkpoints"'"'
        
    # Check if the command succeeded
    if [ $? -eq 0 ]; then
        echo "✅ Successfully completed: $checkpoint with model $model and checkpoint $resume_checkpoints"
    else
        echo "❌ Failed: $checkpoint with model $model and checkpoint $resume_checkpoints"
        echo "Do you want to continue with the next checkpoint? (y/n)"
        read -r response
        if [[ "$response" != "y" && "$response" != "Y" ]]; then
            echo "Stopping execution."
            exit 1
        fi
    fi
    
    echo ""
    echo "Waiting 10 seconds before next combination..."
    sleep 10
done


echo "🎉 All checkpoint consolidation jobs completed!"

# echo "⬇️ Downloading checkpoints from GCS to local..."
# python gpu_utils.py --checkpoint_dirs "${CHECKPOINT_DIRS[@]}" --resume_checkpoints "${RESUME_CHECKPOINTS[@]}"
# echo "✅ All checkpoints downloaded to local."