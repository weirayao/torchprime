XLA_IR_DEBUG=1 XLA_HLO_DEBUG=1 python torchprime/torch_xla_models/inference.py \
    global_batch_size=8 \
    model.tokenizer_name=/home/haolin.chen/sfr-text-diffusion-model-research/consolidated_checkpoints/flex-qwen3-1b-gcs-pretrain-all-data/15000 \
    generation.diffusion_steps=32 \
    generation.max_tokens=256 \
    checkpoint_dir=gs://sfr-text-diffusion-model-research/checkpoints/flex-qwen3-1b-gcs-pretrain-all-data-1024 \
    resume_from_checkpoint=50000 \
    ici_mesh.fsdp=8 \
    ici_mesh.tensor=1 \
    ici_mesh.data=1 \
    ici_mesh.expert=1 \
    model/remat=qwen-scan
