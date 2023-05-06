export WANDB_API_KEY=9caf287e2fa36f9865f8d491ff47c0e1a35ac04e
export WANDB_PROJECT=santacoder-github-commit
SANTACODER_DIR=/dev/cache/qian/checkpoints/santacoder
export HF_MODULES_CACHE=$SANTACODER_DIR
export PYTHONPATH=$PYTHONPATH:$SANTACODER_DIR

deepspeed train.py \
      --max_input_length 2048 \
      --dataset_name bigcode/commits-pjj-2048-dedup \
      --max_steps 100000 \
      --batch_size 8 \
      --gradient_accumulation_steps 1 \
      --learning_rate 5e-5 \
      --num_warmup_steps 1000 \
      --eval_freq 3000 \
      --save_freq 3000 \
      --log_freq 10 \
      --num_workers 8 \
      --bf16 \
      --deepspeed zero_stage1_config.json \
      --cache_dir /dev/cache/qian/datasets \
      --compute_loss_on_input \
      --data_packing \
      --add_file_name \
      --output_dir /dev/cache/qian/checkpoints/santacoder_v11_instruction_dedup