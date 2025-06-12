XLA_IR_DEBUG=1 XLA_HLO_DEBUG=1 python torchprime/torch_xla_models/inference.py \
    model.tokenizer_name=/home/haolin.chen/sfr-text-diffusion-model-research/consolidated_checkpoints/test-consolidation-flex-qwen3-1b-gcs/None \
    model/remat=qwen-scan
