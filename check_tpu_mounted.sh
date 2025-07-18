TPU_VM_NAME="sfr-haolin-chen-v4-1024" # Change with your TPU VM name
TPU_ZONE="us-central2-b"
WORKER="all"

gcloud alpha compute tpus tpu-vm ssh $TPU_VM_NAME \
    --zone=$TPU_ZONE \
    --project=salesforce-research-internal \
    --tunnel-through-iap \
    --worker=$WORKER \
    --command='
        if [ -f /root/sfr-text-diffusion-model-research/data/flex_processed_v1/haolin/proc_0/RedPajama_part0_532d5d9b46a44a75b93a67fed72cebca.parquet ]; then
            echo "pass"
        else
            echo "Worker $(hostname): directory does NOT exist"
        fi
    '