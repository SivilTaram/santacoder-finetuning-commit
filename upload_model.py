import os


def clean_dir(dir_path):
    required_files = ["pytorch_model.bin", "config.json",
                      "tokenizer_config.json", "special_tokens_map.json",
                      "configuration_gpt2_mq.py", "modeling_gpt2_mq.py",
                      "tokenizer.json"]
    for filename in os.listdir(dir_path):  # 遍历目录下的所有文件
        file_path = os.path.join(dir_path, filename)  # 拼接文件路径

        if os.path.isfile(file_path) and filename not in required_files:  # 如果是文件且不是 pytorch_model.bin
            os.remove(file_path)  # 删除文件


def copy_files(dir_path):
    src_dir_path = "/dev/cache/qian/checkpoints/santacoder"
    for filename in os.listdir(src_dir_path):
        file_path = os.path.join(src_dir_path, filename)
        if os.path.isfile(file_path) and filename != 'pytorch_model.bin':
            os.system(f"cp {file_path} {dir_path}")


def cp_dir(src_dir, tgt_dir):
    os.system(f"mv {src_dir} {tgt_dir}")


def mv_subdir(src_dir, tgt_dir):
    for sub_dir in os.listdir(src_dir):
        file_path = os.path.join(src_dir, sub_dir)
        if os.path.isdir(file_path):
            print("Cleaning", file_path)
            clean_dir(file_path)
            print("Moving", file_path)
            cp_dir(file_path, tgt_dir)


def unify_generation_files(dir_path):
    model_name = "santacoder-verb-filter-2048-unpacking"
    os.makedirs(f"download/{model_name}", exist_ok=True)
    fetch_files = ["generations_humanevalxbugspy_greedy.json",
                   "generations_humanevalxbugsjava_greedy.json",
                   "generations_humanevalxbugsjs_greedy.json"]
    for sub_dir in os.listdir(dir_path):
        sub_dir_path = os.path.join(dir_path, sub_dir)
        if os.path.isdir(sub_dir_path):
            # take the file
            for fetch_file in fetch_files:
                fetch_file_path = os.path.join(sub_dir_path, fetch_file)
                target_file_path = os.path.join("download", model_name,
                                                fetch_file.replace(".json", "[{}].json".format(sub_dir)))
                if os.path.isfile(fetch_file_path):
                    os.system(f"cp {fetch_file_path} {target_file_path}")


def unify_evaluation_files(dir_path):
    model_name = "santacoder-verb-filter-2048-unpacking-eval"
    os.makedirs(f"download/{model_name}", exist_ok=True)
    fetch_files = ["evaluation_humanevalxbugspy_greedy.json"]
    for sub_dir in os.listdir(dir_path):
        sub_dir_path = os.path.join(dir_path, sub_dir)
        if os.path.isdir(sub_dir_path):
            # take the file
            for fetch_file in fetch_files:
                fetch_file_path = os.path.join(sub_dir_path, fetch_file)
                target_file_path = os.path.join("download", model_name,
                                                fetch_file.replace(".json", "[{}].json".format(sub_dir)))
                if os.path.isfile(fetch_file_path):
                    os.system(f"cp {fetch_file_path} {target_file_path}")


def clear_generation_files(dir_path):
    fetch_files = ["generations_humanevalxbugspy_greedy.json",
                   "generations_humanevalxbugsjs_greedy.json",
                   "generations_humanevalxbugsjava_greedy.json"]
    for sub_dir in os.listdir(dir_path):
        sub_dir_path = os.path.join(dir_path, sub_dir)
        if os.path.isdir(sub_dir_path):
            # take the file
            for fetch_file in fetch_files:
                fetch_file_path = os.path.join(sub_dir_path, fetch_file)
                if os.path.isfile(fetch_file_path):
                    os.remove(fetch_file_path)


def clear_models(dir_path):
    for sub_dir in os.listdir(dir_path):
        sub_dir_path = os.path.join(dir_path, sub_dir)
        if os.path.isdir(sub_dir_path) and "checkpoint" in sub_dir:
            # remove checkpoint
            sub_dir_name = sub_dir_path.split("-")[-1]
            # if the model cannot be divided by 18000, remove it
            if int(sub_dir_name) % 18000 != 0:
                # delete
                os.system(f"rm -rf {sub_dir_path}")


if __name__ == '__main__':
    # unify_generation_files("/dev/cache/qian/checkpoints/santacoder_v10_instruction_verb_filter_2048_no_packing")
    # unify_evaluation_files("/dev/cache/qian/checkpoints/santacoder_v10_instruction_verb_filter_2048_no_packing")

    clear_generation_files("/dev/cache/qian/checkpoints/santacoder-pjj-2048-unpacking-lawa")
    # mv_subdir("/dev/cache/qian/checkpoints/santacoder_v10_instruction_verb_filter_2048_no_packing_add_file",
    #           "/dev/cache/qian/checkpoints/santacoder-commits-pjj-2048")
    # clear_models("/dev/cache/qian/checkpoints/santacoder-commits-pjj-2048")