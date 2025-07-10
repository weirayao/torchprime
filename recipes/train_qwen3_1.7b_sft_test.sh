# First, create the test dataset
python torchprime/torch_xla_models/create_test_dataset.py

# Then run SFT training with the test dataset
XLA_IR_DEBUG=1 XLA_HLO_DEBUG=1 python torchprime/torch_xla_models/train.py \
    data=sft \
    model=flex-qwen-1b \
    training_mode=sft \
    global_batch_size=8 \
    max_steps=40 \
    checkpoint_dir=gs://sfr-text-diffusion-model-research/checkpoints/test-flex-qwen3-1b-sft-test-gcs \
    resume_from_checkpoint=20 \
    save_steps=20 \
    logging_steps=1 \
    ici_mesh.fsdp=8 \
    ici_mesh.tensor=1 \
    ici_mesh.data=1 \
    ici_mesh.expert=1 \
    model/remat=qwen-scan \
    data.dataset_name=test_sft_dataset \
    data.sft.format=alpaca \
    data.sft.include_system_prompt=true
# fsdp * tensor * data * expert == num_devices
# global_batch_size mod num_devices == 0 