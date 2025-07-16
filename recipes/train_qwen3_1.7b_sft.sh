XLA_IR_DEBUG=1 XLA_HLO_DEBUG=1 python torchprime/torch_xla_models/train.py \
    data=sft \
    model=flex-qwen-1b \
    training_mode=sft \
    global_batch_size=16 \
    max_steps=16010 \
    checkpoint_dir=gs://sfr-text-diffusion-model-research/checkpoints/flex_processed_v1_qw1_7b_512_split_datafix \
    resume_from_checkpoint=16000 \
    sft_save_dir=gs://sfr-text-diffusion-model-research/checkpoints/test-flex-qwen3-1b-sft-gcs_sft \
    save_steps=100 \
    logging_steps=1 \
    ici_mesh.fsdp=16 \
    ici_mesh.tensor=1 \
    ici_mesh.data=1 \
    ici_mesh.expert=1 \
    model/remat=qwen-scan \
    data.dataset_name=tatsu-lab/alpaca \
    data.sft.format=alpaca \
    data.sft.include_system_prompt=true \
    model.attention_kernel=default
# fsdp * tensor * data * expert == num_devices
# global_batch_size mod num_devices == 0 
# 