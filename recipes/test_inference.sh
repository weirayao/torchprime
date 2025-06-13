XLA_IR_DEBUG=1 XLA_HLO_DEBUG=1 python torchprime/torch_xla_models/inference.py \
    model.tokenizer_name=/home/haolin.chen/sfr-text-diffusion-model-research/consolidated_checkpoints/flex-qwen3-1b-gcs-pretrain-all-data/15000 \
    generation.diffusion_steps=32 \
    generation.max_new_tokens=0 \
    checkpoint_dir=gs://sfr-text-diffusion-model-research/checkpoints/flex-qwen3-1b-gcs-pretrain-all-data \
    resume_from_checkpoint=15000 \
    model/remat=qwen-scan
