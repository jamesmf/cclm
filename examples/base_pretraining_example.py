import os
from cclm.preprocessing import Preprocessor
from cclm.pretrainers.cl_mask_pretrainer import (
    CLMaskPretrainer,
    CLMaskPretrainerEvaluationCallback,
)
from cclm.preprocessing import Preprocessor
from cclm.models import Embedder
from cclm.augmentation import Augmentor
from datasets import load_dataset
import tensorflow as tf
import argparse
import mlflow

mlflow.set_tracking_uri("sqlite:///tracking.db")

ap = argparse.ArgumentParser()
ap.add_argument("--load", help="path to load weights from", default=None)
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
    default=3e-5,
    type=float,
)
ap.add_argument(
    "--maxlen",
    help="max example length",
    dest="max_example_len",
    default=256,
    type=int,
)
args = ap.parse_args()

mlflow.log_params(vars(args))
run_info = mlflow.active_run().to_dictionary()["info"]
artifact_path = run_info["artifact_uri"]
run_id = run_info["run_id"]


dataset = load_dataset(
    "wikipedia", "20200501.en", cache_dir="/app/cclm/.datasets"
).shuffle()["train"]

augmentor = Augmentor()

prep = Preprocessor(max_example_len=args.max_example_len)
# load or fit the preprocessor
if args.load:
    prep._load(os.path.join(args.load, "cclm_config.json"))
else:
    prep.fit(
        dataset[:100000]["text"], min_char_count=args.min_char, skip_tokenizer=True
    )

mlflow.log_metric("n_chars", len(prep.char_rev))
# initialize the base and possibly load its embedder
base = Embedder(prep.max_example_len, prep.n_chars, load_from=args.load)

# initialize the pretrainer and optionally load an already trained model
bp = CLMaskPretrainer(embedder=base, augmentor=augmentor, preprocessor=prep)
if args.load:
    bp.model = tf.keras.models.load_model(os.path.join(args.load, "model"))

bp.model.compile(
    tf.keras.optimizers.Adam(lr=args.lr),
    tf.keras.losses.SparseCategoricalCrossentropy(),
)
gen = bp.generator(dataset)
print(bp.model.summary())

callbacks = [
    CLMaskPretrainerEvaluationCallback(dataset, bp),
]

bp.model.fit(gen, steps_per_epoch=5000, epochs=500, callbacks=callbacks)
x, y = next(gen)
print(bp.evaluate_prediction(x, bp.model.predict(x), prep))

os.makedirs(artifact_path, exist_ok=True)
prep.save(artifact_path)
base.save(artifact_path)
bp.model.save(os.path.join(artifact_path, "model"))
