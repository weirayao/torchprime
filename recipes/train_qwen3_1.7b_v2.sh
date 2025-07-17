XLA_IR_DEBUG=1 XLA_HLO_DEBUG=1 python torchprime/torch_xla_models/train.py \
    data=flex_v2 \
    model=flex-qwen-1b \
    model.block_masking_probability=0.5 \
    model.truncate_probability=0.5 \
    model.prefix_probability=0.5 \
    global_batch_size=8 \
    max_steps=100 \
    checkpoint_dir=gs://sfr-text-diffusion-model-research/checkpoints/test-masking-flex-qwen3-1b-v2 \
    resume_from_checkpoint=null \
    save_steps=10 \
    logging_steps=1 \
    ici_mesh.fsdp=8 \
    ici_mesh.tensor=1 \
    ici_mesh.data=1 \
    ici_mesh.expert=1 \
    model/remat=qwen-scan
# fsdp * tensor * data * expert == num_devices
# global_batch_size mod num_devices == 0