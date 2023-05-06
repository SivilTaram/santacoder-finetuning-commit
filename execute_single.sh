
cd evaluation
checkpoint_dir=/dev/cache/liuqian/santacoder_megatron_v2
python main.py --tasks humaneval-x-bugs-python \
--allow_code_execution  \
--trust_remote_code  \
--mutate_method edit  \
--n_samples 1  \
--batch_size 1 \
--generations_path $checkpoint_dir/generations_humanevalxbugspy_greedy.json \
--output_path $checkpoint_dir/evaluation_humanevalxbugspy_greedy.json