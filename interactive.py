import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("bigcode/santacoder")
EOD = "<|endoftext|>"

required_files = ["configuration_gpt2_mq.py", "modeling_gpt2_mq.py"]
SANTACODER_DIR = "/dev/cache/qian/checkpoints/santacoder"
TEST_MODEL_CHECKPOINT = "/dev/cache/qian/checkpoints/santacoder_v10_instruction_strict_filter/checkpoint-24000"

for filename in required_files:
    file_path = os.path.join(TEST_MODEL_CHECKPOINT, filename)
    if not os.path.exists(file_path):
        # copy it from santacoder
        os.system(f"cp {SANTACODER_DIR}/{filename} {TEST_MODEL_CHECKPOINT}")

model = AutoModelForCausalLM.from_pretrained(TEST_MODEL_CHECKPOINT,
                                             device_map="auto",
                                             trust_remote_code=True)

raw_text = """<commit_before>
/* Check if in given list of numbers, are any two numbers closer to each other than
  given threshold.
  >>> hasCloseElements([1.0, 2.0, 3.0], 0.5)
  false
  >>> hasCloseElements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
  true
  */
const hasCloseElements = (numbers, threshold) => {
  for (let i = 0; i < numbers.length; i++) {
    for (let j = 0; j < numbers.length; j++) {
      if (i != j) {
        let distance = numbers[i] - numbers[j];
        if (distance < threshold) {
          return true;
        }
      }
    }
  }
  return false;
}
<commit_msg>
Fix bug in HasCloseElements
<commit_after>
"""
# commit_message = "[postgres] support :login_timeout as a standalone config option"
input_ids = tokenizer(raw_text, return_tensors="pt").input_ids.to("cuda")
outputs = model.generate(input_ids,
                         do_sample=False,
                         # top_p=0.95,
                         # temperature=0,
                         max_length=2048,
                         pad_token_id=tokenizer.pad_token_id)

output = tokenizer.decode(outputs[0])[len(raw_text):]
if EOD in output:
    output = output[:output.find(EOD)]
print(output)

