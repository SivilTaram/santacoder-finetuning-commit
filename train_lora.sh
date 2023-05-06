export WANDB_API_KEY=9caf287e2fa36f9865f8d491ff47c0e1a35ac04e
export WANDB_PROJECT=santacoder-github-commit
SANTACODER_DIR=/dev/cache/qian/checkpoints/santacoder
export HF_MODULES_CACHE=$SANTACODER_DIR
export PYTHONPATH=$PYTHONPATH:$SANTACODER_DIR
export OMP_NUM_THREADS=1

torchrun --nproc_per_node 8 train.py \
      --max_input_length 2048 \
      --dataset_name bigcode/commits-pjj-2048 \
      --max_steps 20000 \
      --batch_size 1 \
      --gradient_accumulation_steps 8 \
      --learning_rate 5e-5 \
      --num_warmup_steps 1000 \
      --eval_freq 3000 \
      --save_freq 3000 \
      --log_freq 10 \
      --num_workers 8 \
      --bf16 \
      --deepspeed zero_stage1_config.json \
      --cache_dir /dev/cache_sail/liuqian/datasets \
      --compute_loss_on_input \
      --data_packing \
      --enable_lora \
      --output_dir /dev/cache/qian/checkpoints/santacoder_v10_instruction_verb_filter_2048_lora