CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --num_machines 1 \
    --num_processes 1 \
    --use_deepspeed \
    --deepspeed_config_file ./ds_configs/zero1.json \
    train.py \
    --bf16 True \
    --model_name_or_path JackFram/llama-160m \
    --data_name_or_path sahil2801/CodeAlpaca-20k \
    --instruction_key instruction \
    --response_key output \
    --use_lora True \
    --max_step 100 \
    --warmup_steps 10 \
    --logging_steps 5 \
    --eval_steps 50 \
    --save_steps 50 \
    --output_dir ./runs/debug
