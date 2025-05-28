export HF_TOKEN="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"
XLA_IR_DEBUG=1 XLA_HLO_DEBUG=1 python torchprime/torch_xla_models/train.py \
    model=qwen-3-1b \
    global_batch_size=8 \
    block_size=4096 \
    max_steps=1000 \
    logging_steps=1 \
    ici_mesh.fsdp=8 \
    ici_mesh.tensor=1 \
    ici_mesh.data=1 \
    ici_mesh.expert=1 \
    model/remat=qwen-scan
# fsdp * tensor * data * expert == global_batch_size
# global_batch_size mod num_devices == 0