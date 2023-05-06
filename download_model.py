from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("bigcode/large-model")

model = AutoModelForCausalLM.from_pretrained("bigcode/large-model")

from datasets import load_dataset

load_dataset("bigcode/commits-pjj-2048", cache_dir="/dev/cache_sail/liuqian/datasets")