XLA_IR_DEBUG=1 XLA_HLO_DEBUG=1 python torchprime/torch_xla_models/train.py \
    data=mixed_train_dataset \
    model=flex-qwen-0_6b \
    global_batch_size=256 \
    max_steps=90000 \
    checkpoint_dir=gs://sfr-text-diffusion-model-research/checkpoints/flex_processed_v1_qw0_6b_corrected \
    save_steps=500 \
    logging_steps=1 \
    ici_mesh.fsdp=256 \
    ici_mesh.tensor=1 \
    ici_mesh.data=1 \
    ici_mesh.expert=1 \
    model/remat=qwen-scan
