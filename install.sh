pip install wandb
pip install regex
pip install pybind11
pip install nltk
pip install accelerate
pip install datasets
pip install evaluate
pip install mosestokenizer
pip install deepspeed
pip install ghdiff
git clone --branch main --single-branch https://github.com/bigcode-project/transformers
cd transformers
pip install .
cd ..
git clone --branch main --single-branch https://github.com/SivilTaram/peft.git
cd peft
pip install .
cd ..
apt-get update
apt-get install -y tmux
apt-get install -y vim
git config --global credential.helper store
export HUGGINGFACE_TOKEN=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
huggingface-cli login