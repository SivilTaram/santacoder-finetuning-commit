num_gpus=$(nvidia-smi -L | wc -l)
echo "num_gpus: $num_gpus"
sed -i '/num_processes:/d' eval_config.yaml
echo "num_processes: $num_gpus" >> eval_config.yaml

cd evaluation

SANTACODER_DIR=/dev/cache/qian/checkpoints/santacoder
MODEL_DIR=/dev/cache/qian/checkpoints/santacoder_v9_100k_instruction_strict_filter

# traverse all checkpoints
for checkpoint_dir in $MODEL_DIR/checkpoint-*; do
    cp $SANTACODER_DIR/config.json $checkpoint_dir/config.json
    cp $SANTACODER_DIR/configuration_gpt2_mq.py $checkpoint_dir/configuration_gpt2_mq.py
    cp $SANTACODER_DIR/modeling_gpt2_mq.py $checkpoint_dir/modeling_gpt2_mq.py
    cp $SANTACODER_DIR/tokenizer.json $checkpoint_dir/tokenizer.json
    cp $SANTACODER_DIR/tokenizer_config.json $checkpoint_dir/tokenizer_config.json
    echo "Evaluating checkpoint: $checkpoint_dir"

    accelerate launch --config_file ../eval_config.yaml main.py \
    --model $checkpoint_dir \
    --tasks parity \
    --temperature 0.7 \
    --do_sample True \
    --n_samples 3200 \
    --batch_size 160 \
    --allow_code_execution \
    --save_generations \
    --trust_remote_code \
    --mutate_method edit \
    --use_auth_token \
    --generations_path $checkpoint_dir/generations_parity_temp07.json \
    --output_path $checkpoint_dir/evaluation_results_parity_temp07.json
done