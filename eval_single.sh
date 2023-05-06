num_gpus=$(nvidia-smi -L | wc -l)
echo "num_gpus: $num_gpus"
sed -i '/num_processes:/d' eval_config.yaml
echo "num_processes: $num_gpus" >> eval_config.yaml

cd evaluation
git pull

SANTACODER_DIR=/dev/cache/qian/checkpoints/santacoder
export HF_MODULES_CACHE=$SANTACODER_DIR
export PYTHONPATH=$PYTHONPATH:$SANTACODER_DIR
MODEL_DIR=/dev/cache/liuqian/santacoder_megatron_v2
sudo chmod -R 777 $MODEL_DIR
sudo chmod -R 777 $SANTACODER_DIR
MUTATE_METHOD=file

export checkpoint_dir=$MODEL_DIR
cp $SANTACODER_DIR/config.json $checkpoint_dir/config.json
cp $SANTACODER_DIR/configuration_gpt2_mq.py $checkpoint_dir/configuration_gpt2_mq.py
cp $SANTACODER_DIR/modeling_gpt2_mq.py $checkpoint_dir/modeling_gpt2_mq.py
cp $SANTACODER_DIR/tokenizer.json $checkpoint_dir/tokenizer.json
cp $SANTACODER_DIR/tokenizer_config.json $checkpoint_dir/tokenizer_config.json

echo "Evaluating checkpoint: $checkpoint_dir"

accelerate launch --config_file ../eval_config.yaml main.py \
--model $checkpoint_dir \
--tasks humaneval-x-bugs-python \
--do_sample False \
--n_samples 1 \
--batch_size 1 \
--save_generations \
--trust_remote_code \
--mutate_method $MUTATE_METHOD \
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
--mutate_method $MUTATE_METHOD \
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
--mutate_method $MUTATE_METHOD \
--generations_path $checkpoint_dir/generations_humanevalxbugsjava_greedy.json \
--generation_only \
--max_length_generation 1024