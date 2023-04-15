export WANDB_API_KEY=9caf287e2fa36f9865f8d491ff47c0e1a35ac04e
export WANDB_PROJECT=santacoder-github-commit

deepspeed train.py \
      --max_input_length 2048 \
      --model_path bigcode/large-model \
      --dataset_name bigcode/commits-pjj-2048 \
      --max_steps 50000 \
      --batch_size 1 \
      --gradient_accumulation_steps 2 \
      --learning_rate 5e-5 \
      --num_warmup_steps 1000 \
      --eval_freq 2000 \
      --save_freq 2000 \
      --log_freq 10 \
      --num_workers 8 \
      --bf16 \
      --deepspeed zero_stage3_config.json \
      --cache_dir /dev/cache_sail/qian/datasets \
      --compute_loss_on_input \
      --data_packing \
      --add_file_name \
      --output_dir /dev/cache/qian/checkpoints/santacoder_v10_instruction_verb_filter_2048_large_model