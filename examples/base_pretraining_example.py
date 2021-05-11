import os
from cclm.preprocessing import Preprocessor
from cclm.pretrainers.base_cl_pretrainer import (
    BasePretrainer,
    BasePretrainerEvaluationCallback,
)
from cclm.preprocessing import Preprocessor
from cclm.models import CCLMModelBase
from cclm.augmentation import Augmentor
from datasets import load_dataset
import tensorflow as tf
import argparse
import mlflow

mlflow.set_tracking_uri("sqlite:///tracking.db")

ap = argparse.ArgumentParser()
ap.add_argument("--load", help="path to load weights from")
ap.add_argument(
    "--min-char",
    help="minimum number of times a character needs to appear to be part of the preprocessor",
    dest="min_char",
    default=200,
    type=int,
)
ap.add_argument(
    "--lr",
    help="learning rate",
    dest="lr",
    default=0.00005,
    type=float,
)
args = ap.parse_args()

mlflow.log_params(vars(args))
run_info = mlflow.active_run().to_dictionary()["info"]
artifact_path = run_info["artifact_uri"]
run_id = run_info["run_id"]


dataset = load_dataset("wikipedia", "20200501.en", cache_dir="/app/cclm/.datasets")[
    "train"
]

augmentor = Augmentor()

prep = Preprocessor()
# load or fit the preprocessor
if args.load:
    prep._load(os.path.join(args.load, "cclm_config.json"))
else:
    prep.fit(
        dataset[:100000]["text"], min_char_count=args.min_char, skip_tokenizer=True
    )

mlflow.log_metric("n_chars", len(prep.char_rev))
# initialize the base and possibly load its embedder
base = CCLMModelBase(max_example_len=prep.max_example_len, n_chars. prep.n_chars)
if args.load:
    base.embedder = tf.keras.models.load_model(os.path.join(args.load, "embedder"))

# initialize the pretrainer and optionally load an already trained model
bp = BasePretrainer(base=base, augmentor=augmentor)
if args.load:
    bp.model = tf.keras.models.load_model(os.path.join(args.load, "model"))

bp.model.compile(
    tf.keras.optimizers.Adam(lr=args.lr),
    tf.keras.losses.SparseCategoricalCrossentropy(),
)
gen = bp.generator(dataset)
print(bp.model.summary())

callbacks = [
    BasePretrainerEvaluationCallback(dataset, bp),
]

bp.model.fit(gen, steps_per_epoch=5000, epochs=500, callbacks=callbacks)
x, y = next(gen)
print(bp.evaluate_prediction(x, bp.model.predict(x)))

os.makedirs(artifact_path, exist_ok=True)
prep.save(artifact_path)
base.embedder.save(os.path.join(artifact_path, "embedder"))
bp.model.save(os.path.join(artifact_path, "model"))
