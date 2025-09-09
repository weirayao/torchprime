#!/bin/bash

TPU_VM_NAME="sfr-haolin-chen-v5p-8"
TPU_ZONE="us-central1-a"
BRANCH="haolin/pretrain_v5p_qwen2"
RECIPE="recipes/ckpt_consolidation.sh"

# Define checkpoint directories and resume checkpoints
GCS_PREFIX="gs://sfr-text-diffusion-model-research/checkpoints/"
CHECKPOINT_DIRS=(
    "pretrain_qwen25_coder_1b_flex_v2_webdataset"
)

RESUME_CHECKPOINTS=\'300,1400\'

for checkpoint_dir in "${CHECKPOINT_DIRS[@]}"; do
    checkpoint="${GCS_PREFIX}${checkpoint_dir}"
    echo "=================================================="
    echo "Running checkpoint consolidation:"
    echo "  Checkpoint dir: $checkpoint"
    echo "  Resume checkpoint: $RESUME_CHECKPOINTS"
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
        bash '"$RECIPE"' '"$checkpoint"' '"$RESUME_CHECKPOINTS"''
        
    # Check if the command succeeded
    if [ $? -eq 0 ]; then
        echo "✅ Successfully completed: $checkpoint with checkpoint $resume_checkpoint"
    else
        echo "❌ Failed: $checkpoint with checkpoint $resume_checkpoint"
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