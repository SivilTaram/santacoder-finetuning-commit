import json
import os
from datasets import load_dataset, DatasetDict
from datasets import concatenate_datasets
from IPython.display import HTML

from tqdm import tqdm
import re
import numpy as np
from markdownify import markdownify as md

ds = load_dataset("HuggingFaceH4/stack-exchange-preferences",
                  split="train",
                  num_proc=32,
                  cache_dir="/dev/cache/liuqian/datasets/stack-exchange-preferences")


def lang_callback(el):
    lang = el['class'][0] if el.has_attr('class') else None

    if not lang is None:
        lang = lang.split("-")[-1]
    return lang


def html2md(text):
    text = md(text, code_language_callback=lang_callback)
    text = re.sub(r"\n\s*\n", "\n\n", text).strip()
    return text.encode('utf-8', 'replace').decode()


def select_best_answer(answers):
    # select the best answer
    select_answers = [answer for answer in answers if answer["selected"]]
    if len(select_answers) > 0:
        select_answers = sorted(select_answers, key=lambda x: x["pm_score"], reverse=True)
    else:
        select_answers = sorted(answers, key=lambda x: x["pm_score"], reverse=True)
    best_answer = select_answers[0]["text"]
    return best_answer


def preprocess(example):
    # initialize empty lists for new samples
    new_examples = {"input": html2md(example["question"]), "output": html2md(select_best_answer(example["answers"]))}
    # construct the samples
    return new_examples


def contains_code(example):
    if "```" in example["input"] or "```" in example["output"]:
        return True
    return False


ds = ds.map(preprocess, num_proc=60)

print("Before filtering: ", len(ds))
ds = ds.filter(contains_code)
print("After filtering: ", len(ds))

ds.push_to_hub("bigcode/code-exchange", private=True)

