#! /bin/bash
TPU_VM_NAME="<your_tpu_vm_name>"
TPU_ZONE="us-central2-b"

# Mount GCS bucket to TPU VM
gcloud alpha compute tpus tpu-vm ssh $TPU_VM_NAME \
    --zone=$TPU_ZONE \
    --project=salesforce-research-internal \
    --tunnel-through-iap \
    --worker=all \
    --command='pkill -f python'