resume_from_checkpoint=(16000 14500 12000 9500 7000 4500 2500)

for checkpoint in "${resume_from_checkpoint[@]}"; do
    accelerate launch --config_file=accelerate_config.yaml evaluation_gpu.py \
        eval_dataset_name_or_path=openai/openai_humaneval \
        eval_batch_size=4 \
        generation.noise_level=0.5 \
        generation.seed=42 \
        generation.diffusion_steps=128 \
        generation.max_tokens=null \
        generation.max_new_tokens=null \
        generation.temperature=0.5 \
        generation.top_p=0.95 \
        generation.top_k=10000 \
        generation.alg=neg_entropy \
        generation.alg_temp=0.2 \
        generation.return_dict_in_generate=true \
        generation.output_history=true \
        checkpoint_dir=flex_processed_v1_qw1_7b_512_split_datafix \
        resume_from_checkpoint=$checkpoint
done