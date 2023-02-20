"""
Fine-Tune SantaCoder on Github commits diff dataset
"""

import argparse
import os
from functools import partial
import difflib
from datasets import load_dataset
from datasets import load_metric
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    DataCollatorWithPadding,
    TrainingArguments,
    logging,
    set_seed,
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="bigcode/santacoder")
    parser.add_argument("--dataset_name", type=str, default="bigcode/github-commits-diff")
    parser.add_argument("--subset", type=str, default="data")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--cache_dir", type=str, default="/tmp/github-commits-diff")
    parser.add_argument("--size_valid_set", type=int, default=5000)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--shuffle_buffer", type=int, default=5000)

    parser.add_argument("--max_input_length", type=int, default=2048)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--message_min_token", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--eos_token_id", type=int, default=49152)

    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--num_warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.05)

    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="/tmp/checkpoints")
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--log_freq", default=1, type=int)
    parser.add_argument("--eval_freq", default=1000, type=int)
    parser.add_argument("--save_freq", default=1000, type=int)
    parser.add_argument("--predict_with_generate", action="store_true")
    return parser.parse_args()


def chars_token_ratio(dataset, tokenizer, data_column, nb_examples=1000):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        total_characters += len(example[data_column])
        total_tokens += len(tokenizer(example[data_column]).tokens())

    return total_characters / total_tokens


def get_too_common_message(dataset, nb_examples=10_0000):
    """
    Perform statistics on the git messages
    """
    common_message = {}
    counter = 0
    for example in dataset:
        message = example["message"]
        if counter > nb_examples:
            break
        if message in common_message:
            common_message[message] += 1
        else:
            common_message[message] = 1
        counter += 1
    common_message = {k: v for k, v in common_message.items() if v > 1000}
    return set(common_message.keys())


def commit_diff(example):
    example["diff"] = "\n".join([line for line in difflib.unified_diff(example["old_contents"].splitlines(),
                                                                       example["new_contents"].splitlines())
                                 if line.startswith("+") or line.startswith("-")][2:])
    return example


tokenizer = AutoTokenizer.from_pretrained("bigcode/santacoder")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.sep_token = tokenizer.eos_token


def preprocess_function(examples, args):
    # input as old content + message, output as code edits
    inputs = [message + " : " + old_content + tokenizer.eos_token + diff_content
              for message, old_content, diff_content in zip(examples["message"],
                                                            examples["old_contents"],
                                                            examples["diff"])]

    model_inputs = tokenizer(inputs,
                             max_length=args.max_input_length,
                             padding="max_length",
                             truncation=True)
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs


def create_datasets(args):
    dataset = load_dataset(
        args.dataset_name,
        data_dir=args.subset,
        split=args.split,
        use_auth_token=True,
        num_proc=args.num_workers if not args.streaming else None,
        streaming=args.streaming,
        cache_dir=args.cache_dir
        # data_files="diffs_55574529_56098816.jsonl"
    )

    chars_per_token_content = chars_token_ratio(dataset, tokenizer, "old_contents")
    chars_per_token_message = chars_token_ratio(dataset, tokenizer, "message")
    print(f"The character to token ratio of the content is: {chars_per_token_content:.2f}")
    print(f"The character to token ratio of the message is: {chars_per_token_message:.2f}")

    common_message = get_too_common_message(dataset)

    # filter all datasets which are not positive
    def valid_filter_dataset(example):
        return example["new_contents"] != example["old_contents"] and \
               len(example["new_contents"].strip()) > 0 and \
               len(example["old_contents"].strip()) > 0 and \
               example["message"] not in common_message and \
               len(example["old_contents"]) < 50_000

    # filter the dataset
    dataset = dataset.filter(valid_filter_dataset,
                             num_proc=args.num_workers)

    dataset = dataset.map(commit_diff,
                          num_proc=args.num_workers)

    # filter all datasets which are not positive
    def length_filter_dataset(example):
        # exclude pull requests message
        return len(example["old_contents"] + example["message"] + example["diff"]) + 1 \
               <= chars_per_token_content * args.max_input_length and \
               len(example["message"]) >= chars_per_token_message * args.message_min_token

    def prefix_filter_dataset(example):
        return not example["message"].startswith("Merge pull request") and \
               not example["message"].startswith("Merge branch") and \
               not example["message"].startswith("Bump version")

    # filter the dataset
    dataset = dataset.filter(length_filter_dataset,
                             num_proc=args.num_workers) \
        .filter(prefix_filter_dataset, num_proc=args.num_workers)

    if args.streaming:
        print("Loading the dataset in streaming mode")
        valid_data = dataset.take(args.size_valid_set)
        train_data = dataset.skip(args.size_valid_set)
        train_data = train_data.shuffle(buffer_size=args.shuffle_buffer, seed=args.seed)
    else:
        dataset = dataset.train_test_split(test_size=0.005, seed=args.seed)
        train_data = dataset["train"]
        valid_data = dataset["test"]
        print(
            f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}"
        )

    column_names = dataset.column_names["train"]
    train_dataset = train_data.map(
        partial(preprocess_function, args=args),
        batched=True,
        num_proc=args.num_workers,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on train dataset",
    )
    valid_dataset = valid_data.map(
        partial(preprocess_function, args=args),
        batched=True,
        num_proc=args.num_workers,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on validation dataset",
    )
    return train_dataset, valid_dataset


def run_training(args, train_data, val_data):
    print("Loading the model")
    # disable caching mechanism when using gradient checkpointing
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_cache=not args.gradient_checkpointing,
    )
    train_data.start_iteration = 0

    print(f"Starting main loop")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        dataloader_drop_last=True,
        evaluation_strategy="steps",
        max_steps=args.max_steps,
        eval_steps=args.eval_freq,
        save_steps=args.save_freq,
        logging_steps=args.log_freq,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.num_warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        fp16=args.fp16,
        bf16=args.bf16,
        deepspeed=args.deepspeed,
        weight_decay=args.weight_decay,
        run_name=f"santacoder-{args.subset}",
        report_to="wandb"
    )
    # Data collator
    data_collator = DataCollatorWithPadding(
        tokenizer,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    metric = load_metric("bleu")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]
        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        print(preds.shape)
        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        return result

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
        data_collator=data_collator,
        # compute_metrics=compute_metrics if args.predict_with_generate else None
    )

    print("Training...")
    trainer.train()

    print("Saving last checkpoint of the model")
    model.save_pretrained(os.path.join(args.output_dir, "final_checkpoint/"))


def main(args):
    train_dataset, eval_dataset = create_datasets(args)
    run_training(args, train_dataset, eval_dataset)


if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    logging.set_verbosity_error()

    main(args)
