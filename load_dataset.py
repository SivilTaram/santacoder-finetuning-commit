from datasets import load_dataset


dataset = load_dataset("SivilTaram/git-commits-cleaned-python-java-javascript")
sampled_dataset = dataset["train"].shuffle().select(range(1000))
# save dataset into json line file
# write_f = open("sampled_message.txt", "w", encoding="utf8")
# for example in sampled_dataset:
#     write_f.write(example["content"].split("<commit_msg>")[1].split("<commit_diff>")[0] + "\n")
sampled_dataset.to_json("sampled_dataset_full.jsonl", orient="records", lines=True)
# write_f.close()