TPU_VM_NAME="sfr-cqin-v4-32-3"
TPU_ZONE="us-central2-b"
BRANCH="cqin/dev"
RECIPE="recipes/train_qwen3_1.7b_sft.sh"
# RECIPE="recipes/train_qwen3_1.7b_sft_256core_config.sh"

# Create logs directory if it doesn't exist
mkdir -p logs

# Default log file name with timestamp
LOG_FILE="logs/log_$(date +%Y%m%d_%H%M%S).log"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -r|--recipe)
      RECIPE="$2"
      shift 2
      ;;
    -l|--log)
      LOG_FILE="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [-r|--recipe RECIPE_PATH] [-l|--log LOG_FILE]"
      echo "  -r, --recipe    Path to training recipe (default: recipes/train_qwen3_1.7b.sh)"
      echo "  -l, --log       Log file name (default: logs/log_YYYYMMDD_HHMMSS.log)"
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

echo "Starting remote run at $(date)" | tee "$LOG_FILE"
echo "Recipe: $RECIPE" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "----------------------------------------" | tee -a "$LOG_FILE"

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
    HYDRA_FULL_ERROR=1 bash '"$RECIPE"'' 2>&1 | tee -a "$LOG_FILE"

echo "----------------------------------------" | tee -a "$LOG_FILE"
echo "Remote run completed at $(date)" | tee -a "$LOG_FILE"
