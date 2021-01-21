"""
Train a basic cclm model

until huggingface tokenizers gets a better API for pretraining:
wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt
"""
from cclm.pretraining import MaskedLanguagePretrainer
from cclm.preprocessing import MLMPreprocessor
from cclm.models import CCLMModelBase
from datasets import load_dataset
import numpy as np
import tensorflow as tf
import pickle
import argparse

ap = argparse.ArgumentParser()
ap.add_argument(
    "--load",
    dest="load_existing",
    help="continue training from .models/ folder",
    action="store_true",
)
args = ap.parse_args()

dataset = load_dataset(
    "wikitext", "wikitext-103-raw-v1", cache_dir="/app/cclm/.datasets"
)
dataset = dataset["train"]["text"]

if not args.load_existing:

    prep = MLMPreprocessor(tokenizer_path=None, max_example_len=512)
    prep.fit(dataset[:100000])
else:
    with open(".models/prep_test.pkl", "rb") as f:
        prep = pickle.load(f)

base = CCLMModelBase(preprocessor=prep)


if args.load_existing:
    pretrainer = MaskedLanguagePretrainer(
        base=base, downsample_factor=16, n_strided_convs=4, load_from=".models/mlm_test"
    )
    base.embedder = tf.keras.models.load_model(".models/mlm_embedder")
else:
    pretrainer = MaskedLanguagePretrainer(
        base=base,
        downsample_factor=16,
        n_strided_convs=4,
    )

pretrainer.fit(dataset[:10000], epochs=5)

# pretrainer.save_model(".models/mlm_test")
# base.save(".models/mlm_embedder")
# with open(".models/prep_test.pkl", "wb") as f:
#     pickle.dump(prep, f)