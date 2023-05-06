"""
Fine-Tune SantaCoder on Github commits diff dataset
"""

import argparse
import os
from functools import partial
from datasets import concatenate_datasets
import torch.cuda
from datasets import load_dataset, Dataset
from datasets import load_metric
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    DataCollatorWithPadding,
    TrainingArguments,
    logging,
    set_seed,
    TrainerCallback,
    AutoConfig
)
import numpy as np
from transformers import AutoModelForSeq2SeqLM
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_int8_training


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="bigcode/santacoder")
    parser.add_argument("--dataset_name", type=str, default="bigcode/instruction-commits")
    # parser.add_argument("--subset", type=str, default="data")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--cache_dir", type=str, default="/tmp/santacoder-github-cleaned")
    parser.add_argument("--size_valid_set", type=int, default=5000)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--shuffle_buffer", type=int, default=25600)

    parser.add_argument("--max_input_length", type=int, default=2048)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--message_min_token", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--eos_token_id", type=int, default=49152)

    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--num_warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.05)

    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--output_diff", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="/tmp/checkpoints")
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--log_freq", default=1, type=int)
    parser.add_argument("--eval_freq", default=1000, type=int)
    parser.add_argument("--save_freq", default=1000, type=int)
    parser.add_argument("--predict_with_generate", action="store_true")
    parser.add_argument("--compute_loss_on_input", default=False, action="store_true")
    parser.add_argument("--compute_loss_on_instruction", default=False, action="store_true")
    parser.add_argument("--data_packing", default=False, action="store_true")
    parser.add_argument("--add_file_name", default=False, action="store_true")
    parser.add_argument("--flan_file_path", type=str, default=None)
    parser.add_argument("--enable_lora", default=False, action="store_true")

    return parser.parse_args()


tokenizer = AutoTokenizer.from_pretrained("bigcode/santacoder")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.sep_token = tokenizer.eos_token


def preprocess_function(examples, args, is_train):
    inputs = examples["input"]
    targets = examples["output"]
    model_inputs = [inp + tar + tokenizer.eos_token for inp, tar in zip(inputs, targets)]

    # do not do packing for fair comparison with no packing
    if args.data_packing and is_train:
        model_inputs = tokenizer(model_inputs,
                                 max_length=args.max_input_length,
                                 # no padding to calculate the real length
                                 padding=False,
                                 truncation=True)
        # packing in the input_ids level
        max_window = args.max_input_length
        # packed_length maintains a list, each of which is a list of all packed sequence lengths in this sequence
        packed_lengths, packed_ids, = [], []
        temp_length, temp_ids = 0, []
        for input_ids in model_inputs["input_ids"]:
            if temp_length + len(input_ids) >= max_window:
                packed_lengths.append(temp_length)
                packed_ids.append(temp_ids.copy())
                # refresh for the next packed sequence
                temp_length = 0
                temp_ids.clear()
            temp_ids.extend(input_ids)
            temp_length += len(input_ids)
        # add the last one
        if temp_length > 0:
            packed_lengths.append(temp_length)
            packed_ids.append(temp_ids.copy())
        # pad the packed_ids into max_window
        padded_input_ids = np.zeros((len(packed_ids), max_window), dtype=np.int64)
        padded_attention_masks = np.zeros((len(packed_ids), max_window), dtype=np.int64)
        for i, example_input_ids in enumerate(packed_ids):
            padded_input_ids[i, :packed_lengths[i]] = example_input_ids
            # obtain the attention mask
            padded_attention_masks[i, :packed_lengths[i]] += 1
        model_inputs["input_ids"] = padded_input_ids
        model_inputs["attention_mask"] = padded_attention_masks
        # take the last valid eos token as the end of the sequence
        first_eos_indices = [packed_lengths[i] - 1 for i in range(len(packed_lengths))]
    else:
        assert tokenizer.eos_token_id == tokenizer.pad_token_id
        model_inputs = tokenizer(model_inputs,
                                 max_length=args.max_input_length,
                                 padding="max_length",
                                 truncation=True)
        # This relies on tokenizer.eos_token_id == tokenizer.pad_token_id
        first_eos_indices = [
            model_inputs["input_ids"][i].index(tokenizer.eos_token_id)
            if model_inputs["input_ids"][i][-1] == tokenizer.eos_token_id else args.max_input_length
            for i in range(len(model_inputs["input_ids"]))
        ]

    if args.compute_loss_on_input:
        assert args.compute_loss_on_instruction is False
        model_inputs["labels"] = [
            list(inp[:first_eos_indices[i] + 1]) + [-100] * (args.max_input_length - first_eos_indices[i] - 1)
            for i, inp in enumerate(model_inputs["input_ids"])
        ]
    else:
        inputs = tokenizer(
            inputs, max_length=args.max_input_length, padding=False, truncation=True
        )["input_ids"]
        # -100 means ignore in loss computation; Make sure only one final eos token is predicted
        model_inputs["labels"] = [
            [-100] * len(inp) + inp_target[len(inp):first_eos_indices[i] + 1] + [-100] * (
                    args.max_input_length - first_eos_indices[i] - 1)
            for i, (inp, inp_target) in enumerate(zip(inputs, model_inputs["input_ids"]))
        ]
    return model_inputs


def create_datasets(args):
    dataset = load_dataset(
        args.dataset_name,
        split=args.split,
        use_auth_token=True,
        num_proc=args.num_workers if not args.streaming else None,
        streaming=args.streaming,
        cache_dir=args.cache_dir
    )

    # if streaming, we only use a small portion of the dataset for debugging
    if args.streaming:
        sample_number = 480
        train_data = []
        for i in range(sample_number):
            train_data.append(next(iter(dataset)))
        dataset = Dataset.from_list(train_data)

    def filter_code(example):
        # filter out the examples that contains python, java and javascript
        if "python" in example["input"] or "java" in example["input"] or "js " in example["input"]:
            return True
        if "python" in example["output"] or "java" in example["output"] or "js " in example["output"]:
            return True
        return False

    print("number of examples before code filtering: ", len(dataset))
    dataset = dataset.filter(filter_code, num_proc=args.num_workers)

    print("number of examples after code filtering: ", len(dataset))
    # filter out the examples that are too long
    dataset = dataset.filter(lambda example: len(example["input"] + example["output"]) < 10 * args.max_input_length,
                             num_proc=args.num_workers)
    print("number of examples after character filtering: ", len(dataset))
    dataset = dataset.filter(lambda example: len(tokenizer(example["input"] + example["output"])["input_ids"]) + 1 < args.max_input_length,
                             num_proc=args.num_workers)
    print("number of examples after token filtering: ", len(dataset))

    dataset = dataset.train_test_split(test_size=0.01, seed=args.seed)

    def unify_format(examples):
        examples["input"] = ["<input>\n" + ins.strip() for ins in examples["input"]]
        examples["output"] = ["\n<output>\n" + ins.strip() for ins in examples["output"]]
        return examples

    if not args.compute_loss_on_input:
        # TODO: currently we do not support data packing when not computing loss on input
        assert args.data_packing is False, "data packing can be used when computing loss on input"

    train_data = dataset["train"].map(unify_format, batched=True, num_proc=args.num_workers)
    valid_data = dataset["test"].map(unify_format, batched=True, num_proc=args.num_workers)
    # if the flan file path is not None, add them into the current train set
    if args.flan_file_path is not None:
        flan_dataset = load_dataset(
            "json",
            data_files=args.flan_file_path,
            use_auth_token=True,
            num_proc=args.num_workers if not args.streaming else None,
            streaming=args.streaming,
            cache_dir=args.cache_dir,
        )
        train_data = concatenate_datasets([train_data, flan_dataset["train"]])
    # shuffle dataset
    train_data = train_data.shuffle(seed=args.seed)
    print(
        f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}"
    )

    column_names = dataset.column_names["train"] + ["input", "output"]
    train_dataset = train_data.map(
        partial(preprocess_function, args=args, is_train=True),
        batched=True,
        num_proc=args.num_workers,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on train dataset",
    )
    valid_dataset = valid_data.map(
        partial(preprocess_function, args=args, is_train=False),
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
        use_cache=not args.gradient_checkpointing
    )

    if args.enable_lora:
        if args.model_path == "bigcode/large-model":
            target_modules = ["c_attn", "q_attn"]
        else:
            target_modules = ["kv_attn", "q_attn"]
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            bias="none",
            target_modules=target_modules,
            lora_dropout=0.1)
        # prepare 8-int model for training
        if args.model_path == "bigcode/large-model":
            model = prepare_model_for_int8_training(model)
        model = get_peft_model(model, config)
        model.print_trainable_parameters()

    print("Model loaded")
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
        run_name=args.output_dir,
        report_to="wandb"
    )
    # Data collator
    data_collator = DataCollatorWithPadding(
        tokenizer,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
        data_collator=data_collator,
        # callbacks=[EvaluateDebugCallback()]
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
