#! /bin/bash
TPU_VM_NAME="sfr-haolin-chen-v5p-16" # Change with your TPU VM name
TPU_ZONE="us-central1-a"
WORKER="all"

# Install python with venv
# gcloud alpha compute tpus tpu-vm ssh $TPU_VM_NAME \
#     --zone=$TPU_ZONE \
#     --project=salesforce-research-internal \
#     --tunnel-through-iap \
#     --worker=$WORKER \
#     --command='
#     sudo apt-get update; \
#     sudo apt-get install python3.10-venv -y'

# # Install torchprime and other dependencies
# gcloud alpha compute tpus tpu-vm ssh $TPU_VM_NAME \
#     --zone=$TPU_ZONE \
#     --project=salesforce-research-internal \
#     --tunnel-through-iap \
#     --worker=$WORKER \
#     --command='
#     cd torchprime; \
#     rm -rf venv; \
#     python3.11 -m venv venv; \
#     source venv/bin/activate; \
#     pip install --upgrade pip setuptools==69.5.1; \
#     pip install torch==2.8.0 torch_xla[tpu]==2.8.0; \
#     pip install --pre torch_xla[pallas] --index-url https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/ --find-links https://storage.googleapis.com/jax-releases/libtpu_releases.html; \
#     pip install -e ".[dev]"; \
#     pip install gcsfs wandb python-dotenv webdataset' > logs/setup_tpu.log 2>&1

# # Install torchprime and other dependencies
# gcloud alpha compute tpus tpu-vm ssh $TPU_VM_NAME \
#     --zone=$TPU_ZONE \
#     --project=salesforce-research-internal \
#     --tunnel-through-iap \
#     --worker=$WORKER \
#     --command='
#     git clone https://github.com/weirayao/torchprime.git; \
#     cd torchprime; \
#     git pull; \
#     python -m venv venv; \
#     source venv/bin/activate; \
#     pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu; \
#     pip install "torch_xla[tpu] @ https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.8.0.dev-cp310-cp310-linux_x86_64.whl" -f https://storage.googleapis.com/libtpu-wheels/index.html; \
#     python -m pip install --upgrade pip; \
#     python -m pip install --upgrade setuptools==69.5.1; \
#     pip install -e ".[dev]"; \
#     pip install gcsfs wandb python-dotenv'

# Install gcsfuse and mount GCS bucket to TPU VM
# gcloud alpha compute tpus tpu-vm ssh $TPU_VM_NAME \
#     --zone=$TPU_ZONE \
#     --project=salesforce-research-internal \
#     --tunnel-through-iap \
#     --worker=$WORKER \
#     --command='
#     sudo apt-get install -y lsb-release; \
#     sudo pkill -9 unattended-upgr || sudo pkill -9 apt-get || true; \
#     export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`; \
#     echo "deb [signed-by=/usr/share/keyrings/cloud.google.asc] https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list; \
#     curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo tee /usr/share/keyrings/cloud.google.asc; \
#     sudo apt-get update; \
#     sudo apt-get install gcsfuse -y; \
#     which gcsfuse || echo "ERROR: gcsfuse not found in PATH"; \
#     mkdir -p ~/sfr-text-diffusion-model-research; \
#     umount ~/sfr-text-diffusion-model-research; \
#     gcsfuse --implicit-dirs --metadata-cache-ttl-secs=86400 --max-conns-per-host=64 sfr-text-diffusion-model-research ~/sfr-text-diffusion-model-research;'

gcloud alpha compute tpus tpu-vm ssh $TPU_VM_NAME \
    --zone=$TPU_ZONE \
    --project=salesforce-research-internal \
    --tunnel-through-iap \
    --worker=$WORKER \
    --command='
    cd torchprime; \
    source venv/bin/activate; \
    pip install webdataset;'
