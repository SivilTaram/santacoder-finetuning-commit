num_gpus=$(nvidia-smi -L | wc -l)
echo "num_gpus: $num_gpus"
sed -i '/num_processes:/d' eval_config.yaml
echo "num_processes: $num_gpus" >> eval_config.yaml

cd evaluation

SANTACODER_DIR=/dev/cache/qian/checkpoints/santacoder
export HF_MODULES_CACHE=$SANTACODER_DIR
export PYTHONPATH=$PYTHONPATH:$SANTACODER_DIR
MODEL_DIR=/dev/cache/qian/checkpoints/santacoder_v9_100k_instruction_strict_filter

# traverse all checkpoints
for checkpoint_dir in $MODEL_DIR/checkpoint-*; do
  echo "Evaluating checkpoint: $checkpoint_dir"

  accelerate launch --config_file ../eval_config.yaml main.py \
  --model $checkpoint_dir \
  --tasks humaneval-x-bugs-python \
  --do_sample False \
  --n_samples 1 \
  --batch_size 1 \
  --save_generations \
  --trust_remote_code \
  --mutate_method edit \
  --generations_path $checkpoint_dir/generations_humanevalxbugspy_greedy.json \
  --generation_only \
  --max_length_generation 1024

  accelerate launch --config_file ../eval_config.yaml main.py \
  --model $checkpoint_dir \
  --tasks humaneval-x-bugs-js \
  --do_sample False \
  --n_samples 1 \
  --batch_size 1 \
  --save_generations \
  --trust_remote_code \
  --mutate_method edit \
  --generations_path $checkpoint_dir/generations_humanevalxbugsjs_greedy.json \
  --generation_only \
  --max_length_generation 1024

  accelerate launch --config_file ../eval_config.yaml main.py \
  --model $checkpoint_dir \
  --tasks humaneval-x-bugs-java \
  --do_sample False \
  --n_samples 1 \
  --batch_size 1 \
  --save_generations \
  --trust_remote_code \
  --mutate_method edit \
  --generations_path $checkpoint_dir/generations_humanevalxbugsjava_greedy.json \
  --generation_only \
  --max_length_generation 1024
done