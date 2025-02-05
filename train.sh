export WANDB_API_KEY=9caf287e2fa36f9865f8d491ff47c0e1a35ac04e
export WANDB_PROJECT=santacoder-github-commit
deepspeed train.py \
      --max_input_length 2048 \
      --dataset_name bigcode/commits-pjj-2048 \
      --max_steps 250000 \
      --batch_size 2 \
      --gradient_accumulation_steps 4 \
      --learning_rate 5e-5 \
      --num_warmup_steps 1000 \
      --eval_freq 10000 \
      --save_freq 10000 \
      --log_freq 10 \
      --num_workers 8 \
      --bf16 \
      --deepspeed zero_stage1_config.json \
      --cache_dir /dev/cache/qian/cache \
      --output_dir /dev/cache/qian/checkpoints/santacoder_v8_300k_instruction_full_0.2