XLA_IR_DEBUG=1 XLA_HLO_DEBUG=1 python torchprime/torch_xla_models/train.py \
    data=mixed_train_dataset \
    model=flex-qwen-0_6b \
    global_batch_size=512 \
    max_steps=480000 \
    checkpoint_dir=gs://sfr-text-diffusion-model-research/checkpoints/flex_processed_v1_qw0_6b \
    save_steps=500 \
    logging_steps=1 \
    ici_mesh.fsdp=256 \
    ici_mesh.tensor=1 \
    ici_mesh.data=1 \
    ici_mesh.expert=1 \
    model/remat=qwen-scan
# fsdp * tensor * data * expert == global_batch_size
# global_batch_size mod num_devices == 0
