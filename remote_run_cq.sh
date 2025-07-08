TPU_VM_NAME="sfr-cqin-v4-16"
TPU_ZONE="us-central2-b"
BRANCH="cqin/dev"
RECIPE="recipes/train_qwen3_1.7b.sh"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -r|--recipe)
      RECIPE="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [-r|--recipe RECIPE_PATH]"
      echo "  -r, --recipe    Path to training recipe (default: recipes/train_qwen3_1.7b.sh)"
      echo "  -h, --help      Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use -h or --help for usage information"
      exit 1
      ;;
  esac
done

gcloud alpha compute tpus tpu-vm ssh $TPU_VM_NAME \
    --zone=$TPU_ZONE \
    --project=salesforce-research-internal \
    --tunnel-through-iap \
    --worker=all \
    --command='
    export HF_HOME="~/huggingface"; \
    export HF_TOKEN=<HF_TOKEN>; \
    export MOUNTED_GCS_DIR="/home/cqin/sfr-text-diffusion-model-research"; \
    cd torchprime; \
    git fetch; \
    git checkout '"$BRANCH"'; \
    git pull; \
    source venv/bin/activate; \
    export WANDB_API_KEY="local-13554988c6f407ff2f10f686b3dc102c7cb7e5e5"; \
    wandb login $WANDB_API_KEY --relogin --host=https://salesforceairesearch.wandb.io; \
    bash '"$RECIPE"'';
