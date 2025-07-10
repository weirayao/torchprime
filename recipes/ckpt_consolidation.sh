XLA_IR_DEBUG=1 XLA_HLO_DEBUG=1 python torchprime/torch_xla_models/ckpt_consolidation.py \
    model=flex-qwen-1b \
    checkpoint_dir=gs://sfr-text-diffusion-model-research/checkpoints/flex_processed_v1_qw1_7b_512_split_datafix \
    resume_from_checkpoint=32000 \
    ici_mesh.fsdp=4 \
    ici_mesh.tensor=1 \
    ici_mesh.data=1 \
    ici_mesh.expert=1 \
    model/remat=qwen-scan
# fsdp * tensor * data * expert == num_devices
# global_batch_size mod num_devices == 0
