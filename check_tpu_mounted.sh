TPU_VM_NAME="sfr-haolin-chen-v5p-128-0" # Change with your TPU VM name
TPU_ZONE="us-central1-a"
WORKER="all"

gcloud alpha compute tpus tpu-vm ssh $TPU_VM_NAME \
    --zone=$TPU_ZONE \
    --project=salesforce-research-internal \
    --tunnel-through-iap \
    --worker=$WORKER \
    --command='
        if [ -f /root/sfr-text-diffusion-model-research/data/flex_v2_consolidated/DM_Mathematics_part0_95656019-6cd7-43c0-99bf-00b0ffdcbb92.parquet ]; then
            echo "pass"
        else
            echo "Worker $(hostname): directory does NOT exist"
        fi
        if [ -f /root/torchprime/venv/bin/activate ]; then
            echo "pass"
        else
            echo "Worker $(hostname): python venv does NOT exist"
        fi
        source /root/torchprime/venv/bin/activate
        python3 -c "import torch_xla" && echo "torch_xla import: pass" || echo "Worker $(hostname): torch_xla import failed"
    '