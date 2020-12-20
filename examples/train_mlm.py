from sys import base_exec_prefix

"""
Train a basic cclm model

until huggingface tokenizers gets a better API for pretraining:
wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt
"""
from cclm.pretraining import MaskedLanguagePretrainer
from cclm.preprocessing import MLMPreprocessor
from cclm.models import CCLMModelBase
from datasets import load_dataset

dataset = load_dataset(
    "wikitext", "wikitext-103-raw-v1", cache_dir="/app/cclm/.datasets"
)
dataset = dataset["train"]["text"][:10000]

prep = MLMPreprocessor(tokenizer_path="/app/cclm/bert-base-uncased-vocab.txt")
prep.fit(dataset)

base = CCLMModelBase(preprocessor=prep)
pretrainer = MaskedLanguagePretrainer(base=base)
