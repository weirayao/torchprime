XLA_IR_DEBUG=1 XLA_HLO_DEBUG=1 python torchprime/torch_xla_models/ckpt_consolidation.py \
    data=wikitext \
    model=flex-qwen-1b \
    global_batch_size=8 \
    max_steps=40 \
    checkpoint_dir=gs://sfr-text-diffusion-model-research/checkpoints/test-flex-qwen3-1b-gcs \
    resume_from_checkpoint=20 \
    save_steps=20 \
    logging_steps=1 \
    ici_mesh.fsdp=8 \
    ici_mesh.tensor=1 \
    ici_mesh.data=1 \
    ici_mesh.expert=1 \
    model/remat=qwen-scan
# fsdp * tensor * data * expert == num_devices
# global_batch_size mod num_devices == 0