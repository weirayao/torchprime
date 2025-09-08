TPU_VM_NAME="sfr-haolin-chen-v5p-1024"
TPU_ZONE="us-central1-a"

# Copy .env file to TPU VM
gcloud alpha compute tpus tpu-vm scp .env $TPU_VM_NAME:~/torchprime/ \
    --zone=$TPU_ZONE \
    --project=salesforce-research-internal \
    --tunnel-through-iap \
    --worker=all
