import json

import pandas as pd
import os


def analyze_sub_dir(target_dirs, root_path):
    pass_rate_dict = {
        "humaneval-x-bugs-python": {},
        "humaneval-x-bugs-java": {},
        "humaneval-x-bugs-js": {}
    }
    language = ["humaneval-x-bugs-python", "humaneval-x-bugs-java", "humaneval-x-bugs-js"]
    for target_dir in target_dirs:
        target_path = os.path.join(root_path, target_dir)
        json_files = os.listdir(target_path)
        # filter with the eval prefix
        json_files = list(filter(lambda x: x.startswith("eval"), json_files))
        for json_file in json_files:
            json_path = os.path.join(target_path, json_file)
            # extract the checkpoint number
            checkpoint_number = json_file.split("-")[1].replace("].json", "")
            json_obj = json.load(open(json_path, "r", encoding="utf8"))
            for lang in language:
                if lang in json_obj:
                    pass_rate_dict[lang][checkpoint_number] = json_obj[lang]["pass@1"]
    # create table
    table = pd.DataFrame(pass_rate_dict)
    # order the table according to the checkpoint number
    table = table.sort_index()
    # print table with \t as separator, and no index
    print(table.to_csv(sep="\t", index=True))


if __name__ == '__main__':
    analyze_sub_dir(["santacoder-verb-filter-2048-unpacking-eval"],
                    "download")