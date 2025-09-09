COMMAND="ps -aux | grep python | grep -v grep |  awk '{print \$2}' | xargs kill -9"
TPU_NAME="sfr-haolin-chen-v5p-1024"

gcloud alpha compute tpus tpu-vm ssh root@$TPU_NAME \
    --zone=us-central1-a \
    --project=salesforce-research-internal \
    --tunnel-through-iap \
    --worker=all \
    --command="$COMMAND"