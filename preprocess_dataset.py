import json
import os
from datasets import load_dataset, DatasetDict
from datasets import concatenate_datasets
from IPython.display import HTML

from tqdm import tqdm
import re
import numpy as np
from markdownify import markdownify as md


ds = load_dataset("HuggingFaceH4/stack-exchange-preferences", split="train", num_proc=16)


def get_chinese_leetcode_instructions(folder_path):
    file_path = "https://huggingface.co/datasets/BAAI/COIG/blob/main/leetcode_instructions.jsonl"
    # download file into local
    os.system(f"wget {file_path}")
    # move it into the right place
    os.system(f"mv leetcode_instructions.jsonl {folder_path}/leetcode_instructions.jsonl")


def get_english_leetcode_instructions(folder_path):
    pass


def get_docstring_dataset():
    pass


def get_stackexchange_dataset():
    pass