XLA_IR_DEBUG=1 XLA_HLO_DEBUG=1 python torchprime/torch_xla_models/train.py \
    data=mixed_train_dataset \
    model=flex-qwen-1b \
    global_batch_size=512 \
    max_steps=45000 \
    checkpoint_dir=gs://sfr-text-diffusion-model-research/checkpoints/flex_processed_v1_qw1_7b_512_split \
    save_steps=500 \
    logging_steps=1 \
    ici_mesh.fsdp=512 \
    ici_mesh.tensor=1 \
    ici_mesh.data=1 \
    ici_mesh.expert=1 \
    resume_from_checkpoint=9500 \
    model/remat=qwen-scan
# fsdp * tensor * data * expert == global_batch_size
# global_batch_size mod num_devices == 0
