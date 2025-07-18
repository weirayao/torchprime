XLA_IR_DEBUG=1 XLA_HLO_DEBUG=1 python torchprime/torch_xla_models/train.py \
    training_mode=pretrain \
    data=flex_v2 \
    model=flex-qwen-1b \
    model.block_masking_probability=0.01 \
    model.mask_block_sizes=[2, 4] \
    model.truncate_probability=0.01 \
    model.prefix_probability=0.01 \
    optimizer.learning_rate=3e-4 \
    global_batch_size=512 \
    max_steps=50000 \
    checkpoint_dir=gs://sfr-text-diffusion-model-research/checkpoints/flex-qwen3-1b-v2 \
    resume_from_checkpoint=null \
    save_steps=500 \
    logging_steps=1 \
    ici_mesh.fsdp=512 \
    ici_mesh.tensor=1 \
    ici_mesh.data=1 \
    ici_mesh.expert=1 \
    model/remat=qwen-scan
# fsdp * tensor * data * expert == num_devices
# global_batch_size mod num_devices == 0