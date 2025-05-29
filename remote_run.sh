# gcloud alpha compute tpus tpu-vm ssh sfr-weiran-yao-v4-512 \
#  --zone=us-central2-b \
#  --project=salesforce-research-internal \
#  --tunnel-through-iap \
#  --worker=all \
#  --command='cd md4; \
#             export PYTHONPATH="$PYTHONPATH:~/md4"; \
#             python md4/main.py --config=md4/configs/md4/openwebtext.py --sharded=true --workdir=./expt'

# TPU_VM_NAME="sfr-haolin-chen-v4-8"
# TPU_ZONE="us-central2-b"

# TPU_VM_NAME="sfr-weiran-yao-v4-512"
# TPU_ZONE="us-central2-b"

# gcloud alpha compute tpus tpu-vm ssh $TPU_VM_NAME \
#     --zone=$TPU_ZONE \
#     --project=salesforce-research-internal \
#     --tunnel-through-iap \
#     --worker=all \
#     --command='ls md4'


TPU_VM_NAME="sfr-haolin-chen-v4-16"
TPU_ZONE="us-central2-b"

# Default recipe if none is provided
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
    export HF_TOKEN="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"; \
    cd torchprime; \
    git checkout haolin/dev; \
    git pull; \
    source venv/bin/activate; \
    bash '"$RECIPE"'
