#! /bin/bash
TPU_VM_NAME="sfr-haolin-chen-v4-16"
TPU_ZONE="us-central2-b"

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
    export HF_TOKEN="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";'

    # export WANDB_API_KEY="local-30ce520f53046710aadc8e519b31ded23bba904c"; \
    # wandb login $WANDB_API_KEY --relogin --host=https://salesforceairesearch.wandb.io'
