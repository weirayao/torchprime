XLA_IR_DEBUG=1 XLA_HLO_DEBUG=1 python torchprime/torch_xla_models/train.py \
    data=gcs_test \
    model=flex-qwen-1b \
    global_batch_size=512 \
    max_steps=240000 \
    checkpoint_dir=gs://sfr-text-diffusion-model-research/checkpoints/flex-qwen3-1b-gcs-pretrain-all-data-dataloader-fix-1024 \
    save_steps=5000 \
    logging_steps=1 \
    ici_mesh.fsdp=512 \
    ici_mesh.tensor=1 \
    ici_mesh.data=1 \
    ici_mesh.expert=1 \
    model/remat=qwen-scan
# fsdp * tensor * data * expert == global_batch_size
# global_batch_size mod num_devices == 0
