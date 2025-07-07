XLA_IR_DEBUG=1 XLA_HLO_DEBUG=1 python torchprime/torch_xla_models/inference.py \
    global_batch_size=8 \
    model.tokenizer_name=Qwen/Qwen3-1.7B \
    generation.diffusion_steps=10 \
    generation.max_tokens=0 \
    generation.temperature=0.2 \
    generation.top_p=0.95 \
    generation.alg=neg_entropy \
    generation.alg_temp=0.2 \
    generation.output_history=true \
    generation.return_dict_in_generate=true \
    checkpoint_dir=gs://sfr-text-diffusion-model-research/checkpoints/flex-qwen3-1b-gcs-pretrain-all-data-dataloader-fix-1024 \
    resume_from_checkpoint=10000 \
    ici_mesh.fsdp=8 \
    ici_mesh.tensor=1 \
    ici_mesh.data=1 \
    ici_mesh.expert=1 \
    model/remat=qwen-scan
