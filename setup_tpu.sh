#! /bin/bash
TPU_VM_NAME="sfr-haolin-chen-v5p-8" # Change with your TPU VM name
BRANCH="haolin/pretrain_v5p_qwen2"
TPU_ZONE="us-central1-a"
WORKER="all"

# Install python with venv
gcloud alpha compute tpus tpu-vm ssh $TPU_VM_NAME \
    --zone=$TPU_ZONE \
    --project=salesforce-research-internal \
    --tunnel-through-iap \
    --worker=$WORKER \
    --command='
    sudo apt-get update; \
    sudo apt-get install python3.11-venv -y'

# Install torchprime and other dependencies
gcloud alpha compute tpus tpu-vm ssh $TPU_VM_NAME \
    --zone=$TPU_ZONE \
    --project=salesforce-research-internal \
    --tunnel-through-iap \
    --worker=$WORKER \
    --command='
    git clone https://github.com/weirayao/torchprime.git; \
    cd torchprime; \
    git fetch; \
    git checkout '"$BRANCH"'; \
    git pull; \
    python3.11 -m venv venv; \
    source venv/bin/activate; \
    pip install --upgrade pip setuptools==69.5.1; \
    pip install torch==2.8.0 torch_xla[tpu]==2.8.0; \
    pip install --pre torch_xla[pallas] --index-url https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/ --find-links https://storage.googleapis.com/jax-releases/libtpu_releases.html; \
    pip install -e ".[dev]"; \
    pip install gcsfs wandb python-dotenv webdataset'


# Install gcsfuse and mount GCS bucket to TPU VM
gcloud alpha compute tpus tpu-vm ssh $TPU_VM_NAME \
    --zone=$TPU_ZONE \
    --project=salesforce-research-internal \
    --tunnel-through-iap \
    --worker=$WORKER \
    --command='
    sudo apt-get install -y lsb-release; \
    sudo pkill -9 unattended-upgr || sudo pkill -9 apt-get || true; \
    export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`; \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.asc] https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list; \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo tee /usr/share/keyrings/cloud.google.asc; \
    sudo apt-get update; \
    sudo apt-get install gcsfuse -y; \
    which gcsfuse || echo "ERROR: gcsfuse not found in PATH"; \
    mkdir -p ~/sfr-text-diffusion-model-research; \
    umount ~/sfr-text-diffusion-model-research; \
    gcsfuse --implicit-dirs --metadata-cache-ttl-secs=86400 --max-conns-per-host=64 sfr-text-diffusion-model-research ~/sfr-text-diffusion-model-research;'
