TPU_VM_NAME="<your_tpu_vm_name>"
TPU_ZONE="us-central2-b"

# Copy .env file to TPU VM
gcloud alpha compute tpus tpu-vm scp .env $TPU_VM_NAME:~/torchprime/ \
    --zone=$TPU_ZONE \
    --project=salesforce-research-internal \
    --tunnel-through-iap \
    --worker=all
