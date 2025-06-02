export HF_TOKEN="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"
XLA_IR_DEBUG=1 XLA_HLO_DEBUG=1 python torchprime/torch_xla_models/debug_data.py \
    model=qwen-3-1b \
    global_batch_size=8 \
    block_size=4096 \
    max_steps=30 \
    checkpoint_dir=gs://sfr-text-diffusion-model-research/checkpoints/test-qwen3-1b \
    checkpoint_step=20 \
    save_steps=10 \
    logging_steps=1 \
    ici_mesh.fsdp=4 \
    ici_mesh.tensor=1 \
    ici_mesh.data=2 \
    ici_mesh.expert=1 \
    model/remat=qwen-scan
# fsdp * tensor * data * expert == global_batch_size
# global_batch_size mod num_devices == 0