#!/bin/bash

TPU_VM_NAME="sfr-haolin-chen-v4-8"
TPU_ZONE="us-central2-b"
BRANCH="haolin/masking_strategy"
RECIPE="recipes/ckpt_consolidation.sh"

# Define checkpoint directories and resume checkpoints
CHECKPOINT_DIRS=(
    # "gs://sfr-text-diffusion-model-research/checkpoints/flex_processed_v1_qw1_7b_512_split_datafix"
    "gs://sfr-text-diffusion-model-research/checkpoints/flex_processed_v1_qw1_7b_512_split_datafix_from_hf"
)

RESUME_CHECKPOINTS=(
    # "16000"
    # "14500"
    "12000"
    # "9500"
    # "7000"
    # "4500"
    # "2500"
)

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -r|--recipe)
      RECIPE="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [-r|--recipe RECIPE_PATH]"
      echo "  -r, --recipe    Path to training recipe (default: recipes/ckpt_consolidation.sh)"
      echo "  -h, --help      Show this help message"
      echo ""
      echo "This script will run checkpoint consolidation for all combinations of:"
      echo "Checkpoint directories: ${CHECKPOINT_DIRS[*]}"
      echo "Resume checkpoints: ${RESUME_CHECKPOINTS[*]}"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use -h or --help for usage information"
      exit 1
      ;;
  esac
done

# Run checkpoint consolidation for each combination
total_combinations=$((${#CHECKPOINT_DIRS[@]} * ${#RESUME_CHECKPOINTS[@]}))
current_combination=0

for checkpoint_dir in "${CHECKPOINT_DIRS[@]}"; do
    for resume_checkpoint in "${RESUME_CHECKPOINTS[@]}"; do
        ((current_combination++))
        echo "=================================================="
        echo "Running checkpoint consolidation [$current_combination/$total_combinations]:"
        echo "  Checkpoint dir: $checkpoint_dir"
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
            bash '"$RECIPE"' "'"$checkpoint_dir"'" "'"$resume_checkpoint"'"'
        
        # Check if the command succeeded
        if [ $? -eq 0 ]; then
            echo "✅ Successfully completed: $checkpoint_dir with checkpoint $resume_checkpoint"
        else
            echo "❌ Failed: $checkpoint_dir with checkpoint $resume_checkpoint"
            echo "Do you want to continue with the next combination? (y/n)"
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
done

echo "🎉 All checkpoint consolidation jobs completed!"
