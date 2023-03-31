import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("bigcode/santacoder-git-commits-python-java-javascript")
EOD = "<|endoftext|>"

model = AutoModelForCausalLM.from_pretrained("bigcode/santacoder-git-commits-python-java-javascript",
                                             device_map="auto",
                                             trust_remote_code=True)
file_input = """ArJdbc::ConnectionMethods.module_eval do
  def postgresql_connection(config)
    begin
      require \'jdbc/postgres\'
      ::Jdbc::Postgres.load_driver(:require) if defined?(::Jdbc::Postgres.load_driver)
    rescue LoadError # assuming driver.jar is on the class-path
    end
end"""
commit_message = "[postgres] support :login_timeout as a standalone config option"
raw_text = """<commit_before>{}<commit_msg>{}<commit_after>""".format(file_input, commit_message)
input_ids = tokenizer(raw_text, return_tensors="pt").input_ids.to("cuda")
outputs = model.generate(input_ids,
                         do_sample=True,
                         top_p=0.95,
                         temperature=0.4,
                         max_length=2048,
                         pad_token_id=tokenizer.pad_token_id)

output = tokenizer.decode(outputs[0])[len(raw_text):]
if EOD in output:
    output = output[:output.find(EOD)]
print(output)

