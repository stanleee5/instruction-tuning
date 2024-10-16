export DEVICES=0,1,2,3
export N_GPU=$(echo $DEVICES | tr ',' '\n' | wc -l)

export MODEL=facebook/opt-1.3b
export DATA=m-a-p/Code-Feedback

set -ex;
CUDA_VISIBLE_DEVICES=$DEVICES torchrun --nproc_per_node $N_GPU \
    --master_port 29520 \
    train.py \
    --deepspeed ./configs/deepspeed/zero1_comm.json \
    --bf16 True \
    --tf32 True \
    --model_name_or_path $MODEL \
    --use_lora False \
    --test_size 128 \
    --data_name_or_path $DATA \
    --dataset_num_proc 16 \
    --conversations_key messages \
    --prompt_template chat \
    --optim adafactor \
    --lr_scheduler_type constant_with_warmup \
    --gradient_accumulation_steps 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --dataset_num_proc 16 \
    --max_seq_length 2048 \
    --learning_rate 1e-5 \
    --max_grad_norm 0.5 \
    --max_step 50 \
    --warmup_steps 10 \
    --logging_steps 5 \
    --eval_steps 20 \
    --save_steps 1000 \
    --report_to tensorboard \
    --output_dir ./runs/opt-1.3b/Code-Feedback
