checkpoint_dir=flex-qwen3-1b-v2
resume_from_checkpoint=(5000 10000 15000 20000 25000 30000 35000 40000 42500 45000 49500)
dataset_name=openai/openai_humaneval

for checkpoint in "${resume_from_checkpoint[@]}"; do
    accelerate launch --config_file=accelerate_config.yaml evaluation_gpu.py \
        eval_dataset_name_or_path=$dataset_name \
        eval_batch_size=4 \
        noise_levels="[0.1,0.25,0.5,0.75]" \
        repeats=3 \
        seed=42 \
        generation.diffusion_steps=128 \
        generation.max_tokens=null \
        generation.max_new_tokens=null \
        generation.temperature=0.5 \
        generation.top_p=0.95 \
        generation.top_k=10000 \
        generation.alg=neg_entropy \
        generation.alg_temp=0.2 \
        generation.return_dict_in_generate=false \
        generation.output_history=false \
        checkpoint_dir=$checkpoint_dir \
        resume_from_checkpoint=$checkpoint \
        > logs/evaluation_$(basename ${dataset_name})_${checkpoint_dir}_${checkpoint}.log 2>&1
done
