TPU_VM_NAME="sfr-haolin-chen-v5p-128-0" # Change with your TPU VM name
TPU_ZONE="us-central1-a"
BRANCH="haolin/pretrain_qwen25_coder_hyperparam_tuning"
RECIPE="recipes/train_qwen2_pretrain_128_seg_attn_8192_context.sh"

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
    cd torchprime; \
    git fetch; \
    git checkout '"$BRANCH"'; \
    git pull; \
    source venv/bin/activate; \
    bash '"$RECIPE"'';
# python torchprime/torch_xla_models/test_segment_ids_attention.py'
