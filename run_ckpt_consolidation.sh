#!/bin/bash

TPU_VM_NAME="sfr-haolin-chen-v4-8"
TPU_ZONE="us-central2-b"
BRANCH="haolin/masking_strategy"
RECIPE="recipes/ckpt_consolidation.sh"

# Define checkpoint directories and resume checkpoints
GCS_PREFIX="gs://sfr-text-diffusion-model-research/checkpoints/"
CHECKPOINT_DIRS=(
    "flex-qwen3-1b-v2"
)

RESUME_CHECKPOINTS=(
    45000
    47500
    49500
)


# Run checkpoint consolidation for each combination
total_combinations=$((${#CHECKPOINT_DIRS[@]} * ${#RESUME_CHECKPOINTS[@]}))
current_combination=0

for checkpoint_dir in "${CHECKPOINT_DIRS[@]}"; do
    for resume_checkpoint in "${RESUME_CHECKPOINTS[@]}"; do
        ((current_combination++))
        checkpoint="${GCS_PREFIX}${checkpoint_dir}"
        echo "=================================================="
        echo "Running checkpoint consolidation [$current_combination/$total_combinations]:"
        echo "  Checkpoint dir: $checkpoint"
        echo "  Resume checkpoint: $resume_checkpoint"
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
            bash '"$RECIPE"' "'"$checkpoint"'" "'"$resume_checkpoint"'"'
        
        # Check if the command succeeded
        if [ $? -eq 0 ]; then
            echo "✅ Successfully completed: $checkpoint with checkpoint $resume_checkpoint"
        else
            echo "❌ Failed: $checkpoint with checkpoint $resume_checkpoint"
        fi
        
        echo ""
        echo "Waiting 10 seconds before next combination..."
        sleep 10
    done
done

echo "🎉 All checkpoint consolidation jobs completed!"

echo "⬇️ Downloading checkpoints from GCS to local..."
python gpu_utils.py --checkpoint_dirs "${CHECKPOINT_DIRS[@]}" --resume_checkpoints "${RESUME_CHECKPOINTS[@]}"
echo "✅ All checkpoints downloaded to local."