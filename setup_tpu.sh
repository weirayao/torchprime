#! /bin/bash
TPU_VM_NAME="sfr-cqin-v4-32-1" # Change with your TPU VM name
TPU_ZONE="us-central2-b"

# Install python with venv
gcloud alpha compute tpus tpu-vm ssh $TPU_VM_NAME \
    --zone=$TPU_ZONE \
    --project=salesforce-research-internal \
    --tunnel-through-iap \
    --worker=all \
    --command='
    sudo apt-get update; \
    sudo apt-get install python3.10-venv -y'

# Install torchprime and other dependencies
gcloud alpha compute tpus tpu-vm ssh $TPU_VM_NAME \
    --zone=$TPU_ZONE \
    --project=salesforce-research-internal \
    --tunnel-through-iap \
    --worker=all \
    --command='
    git clone https://github.com/weirayao/torchprime.git; \
    cd torchprime; \
    git pull; \
    python -m venv venv; \
    source venv/bin/activate; \
    pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu; \
    pip install "torch_xla[tpu] @ https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.8.0.dev-cp310-cp310-linux_x86_64.whl" -f https://storage.googleapis.com/libtpu-wheels/index.html; \
    python -m pip install --upgrade pip; \
    python -m pip install --upgrade setuptools==69.5.1; \
    pip install -e ".[dev]"; \
    pip install gcsfs wandb python-dotenv'

# Install wandb
gcloud alpha compute tpus tpu-vm ssh $TPU_VM_NAME \
    --zone=$TPU_ZONE \
    --project=salesforce-research-internal \
    --tunnel-through-iap \
    --worker=all \
    --command='
    cd torchprime; \
    source venv/bin/activate; \
    pip install wandb; \
    wandb login $WANDB_API_KEY --relogin --host=https://salesforceairesearch.wandb.io'

# Install gcsfuse and mount GCS bucket to TPU VM
gcloud alpha compute tpus tpu-vm ssh $TPU_VM_NAME \
    --zone=$TPU_ZONE \
    --project=salesforce-research-internal \
    --tunnel-through-iap \
    --worker=all \
    --command='
    export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`; \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.asc] https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list; \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo tee /usr/share/keyrings/cloud.google.asc; \
    sudo apt-get update; \
    sudo apt-get install gcsfuse -y; \
    mkdir -p ~/sfr-text-diffusion-model-research; \
    umount ~/sfr-text-diffusion-model-research; \
    gcsfuse --implicit-dirs --metadata-cache-ttl-secs 0 sfr-text-diffusion-model-research ~/sfr-text-diffusion-model-research;'
