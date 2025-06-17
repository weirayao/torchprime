#! /bin/bash
TPU_VM_NAME="sfr-cqin-v4-16"
TPU_ZONE="us-central2-b"

# Clean up torchprime directory and unmount GCS bucket
gcloud alpha compute tpus tpu-vm ssh $TPU_VM_NAME \
    --zone=$TPU_ZONE \
    --project=salesforce-research-internal \
    --tunnel-through-iap \
    --worker=all \
    --command='
    # Kill any running Python processes
    # pkill -f python
    
    # Unmount GCS bucket
    # umount ~/sfr-text-diffusion-model-research
    
    # Remove torchprime directory and its contents
    rm -rf ~/torchprime
    
    # Remove virtual environment
    # rm -rf ~/torchprime/venv
    
    # Remove GCS mount point
    # rm -rf ~/sfr-text-diffusion-model-research
    
    echo "Cleanup completed successfully"
' 