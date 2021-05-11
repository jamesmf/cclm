"""
Evaluate the idea of a ComposedModel by testing 
- a ComposedModel with two not-pretrained initial models
vs 
- the same two models once pretrained

"""
import os
import argparse
from cclm.pretrainers import MaskedLanguagePretrainer
from cclm.preprocessing import Preprocessor
from cclm.models import CCLMModelBase, ComposedModel
from datasets import load_dataset
import numpy as np
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import mlflow

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="ag_news", dest="dataset")
ap.add_argument(
    "--num-examples",
    default=None,
    dest="num_examples",
    help="number of examples to train on from each dataset",
)
ap.add_argument(
    "--skip-pretrain",
    action="store_true",
    dest="skip_pretrain",
    help="whether to pretrain on another dataset",
)
ap.add_argument("--lr", dest="lr", help="learning rate", type=float, default=0.001)
ap.add_argument(
    "--load-base",
    dest="load_base",
    help="path to pretrained base and pretrainer",
    type=str,
)
args = ap.parse_args()

mlflow.set_tracking_uri("sqlite:///tracking.db")
mlflow.tensorflow.autolog(every_n_iter=1)

mlflow.log_params(vars(args))

# consider "yahoo_answers_topics"

policy = mixed_precision.Policy("mixed_float16")
mixed_precision.set_policy(policy)

# get AG News dataset as an example dataset
dataset = load_dataset(args.dataset, cache_dir="/app/cclm/.datasets")
dataset_train = dataset["train"]["text"]
if args.num_examples is not None:
    dataset_train = dataset_train[: args.num_examples]
dataset_test = dataset["test"]["text"]
y_train = tf.keras.utils.to_categorical(dataset["train"]["label"])
y_test = tf.keras.utils.to_categorical(dataset["test"]["label"])

# create the preprocessor and fit it on the training set
if args.load_base is None:
    prep = Preprocessor(max_example_len=128, vocab_size=10000)
    prep.fit(dataset_train)
else:
    prep = Preprocessor()
    prep._load(args.load_base)


x_train = np.array(
    [prep.string_to_array(i, prep.max_example_len) for i in dataset_train]
)
x_test = np.array([prep.string_to_array(i, prep.max_example_len) for i in dataset_test])

# create a base that embeds the input
if args.load_base is None:
    base = CCLMModelBase(prep.max_example_len, prep.n_chars)
else:
    base = CCLMModelBase(prep.max_example_len, prep.n_chars)
    base.embedder = tf.keras.models.load_model(os.path.join(args.load_base, "embedder"))
    base.embedder.trainable = False

# create two models that we'll combine - optionally we can pretrain one or more of them

pretrainer_a = MaskedLanguagePretrainer(
    base=base,
    downsample_factor=1,
    n_strided_convs=4,
    stride_len=1,
)

pretrainer_b = MaskedLanguagePretrainer(
    base=base,
    downsample_factor=16,
    n_strided_convs=4,
)


def custom_loss(y, pred):
    clip = tf.clip_by_value(pred, 0.01, 0.8)
    return tf.losses.categorical_crossentropy(y, clip)


if not args.skip_pretrain:
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=".tensorboard", histogram_freq=1
    )
    pretraining_generator = pretrainer_a.generator(dataset_train, batch_size=16)
    pretrainer_a.pretraining_model.compile(
        tf.keras.optimizers.Adam(args.lr),
        custom_loss,
        metrics=["categorical_accuracy"],
    )
    pretrainer_a.pretraining_model.fit(
        pretraining_generator,
        epochs=50,
        steps_per_epoch=5000,
        callbacks=[tensorboard_callback],
    )

print(pretrainer_a.model.summary())
print(pretrainer_a.base.embedder.summary())

composed2 = ComposedModel(base, [pretrainer_a.model, pretrainer_b.model])

# put a classification head on it
gmp = tf.keras.layers.GlobalMaxPool1D()
d = tf.keras.layers.Dense(4)
out = tf.keras.layers.Activation("softmax", dtype="float32")
pretrained = tf.keras.Model(composed2.model.input, out(d(gmp(composed2.model.output))))
pretrained.compile(
    tf.keras.optimizers.Adam(0.0005), "categorical_crossentropy", metrics=["accuracy"]
)
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=".tensorboard", histogram_freq=1
)
history_pretrained = pretrained.fit(
    x_train,
    y_train,
    validation_data=(x_test, y_test),
    epochs=15,
    batch_size=32,
    callbacks=[tensorboard_callback],
)
