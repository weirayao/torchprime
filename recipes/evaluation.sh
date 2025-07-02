XLA_IR_DEBUG=1 XLA_HLO_DEBUG=1 python torchprime/torch_xla_models/evaluation.py \
    global_batch_size=8 \
    model.tokenizer_name=/home/haolin.chen/sfr-text-diffusion-model-research/consolidated_checkpoints/flex-qwen3-1b-gcs-pretrain-all-data/15000 \
    generation.diffusion_steps=10 \
    generation.max_tokens=null \
    generation.max_new_tokens=null \
    generation.temperature=0 \
    generation.top_p=0.95 \
    generation.top_k=null \
    eval_dataset_name_or_path=loubnabnl/humaneval_infilling \
    eval_results_save_path=evaluations \
    checkpoint_dir=gs://sfr-text-diffusion-model-research/checkpoints/flex_processed_v1_qw1_7b \
    resume_from_checkpoint=17000 \
    ici_mesh.fsdp=8 \
    ici_mesh.tensor=1 \
    ici_mesh.data=1 \
    ici_mesh.expert=1 \
    model/remat=qwen-scan
