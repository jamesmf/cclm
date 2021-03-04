"""
Evaluate the idea of a ComposedModel by testing 
- a ComposedModel with two not-pretrained initial models
vs 
- the same two models once pretrained

"""
from cclm.pretraining import MaskedLanguagePretrainer
from cclm.preprocessing import MLMPreprocessor
from cclm.models import CCLMModelBase, ComposedModel
from datasets import load_dataset
import numpy as np
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision


policy = mixed_precision.Policy("mixed_float16")
mixed_precision.set_policy(policy)

# get AG News dataset as an example dataset
dataset = load_dataset("ag_news", cache_dir="/app/cclm/.datasets")
dataset_train = dataset["train"]["text"]
dataset_test = dataset["test"]["text"]
y_train = tf.keras.utils.to_categorical(dataset["train"]["label"])
y_test = tf.keras.utils.to_categorical(dataset["test"]["label"])

# create the preprocessor and fit it on the training set
prep = MLMPreprocessor(max_example_len=1024)
prep.fit(dataset_train)

x_train = np.array(
    [prep.string_to_array(i, prep.max_example_len) for i in dataset_train]
)
x_test = np.array([prep.string_to_array(i, prep.max_example_len) for i in dataset_test])

# # create a base
# base = CCLMModelBase(preprocessor=prep)

# # create two pretrainers that we'll combine
# pretrainer_a = MaskedLanguagePretrainer(
#     base=base,
#     downsample_factor=16,
#     n_strided_convs=4,
# )

# pretrainer_b = MaskedLanguagePretrainer(
#     base=base,
#     downsample_factor=16,
#     n_strided_convs=4,
# )

# composed = ComposedModel(base, [pretrainer_a.model, pretrainer_b.model])

# # put a classification head on it
# gmp = tf.keras.layers.GlobalMaxPool1D()
# d = tf.keras.layers.Dense(4)
# out = tf.keras.layers.Activation("softmax", dtype="float32")
# not_pretrained = tf.keras.Model(
#     composed.model.input, out(d(gmp(composed.model.output)))
# )
# not_pretrained.compile(
#     tf.keras.optimizers.Adam(0.0001), "categorical_crossentropy", metrics=["accuracy"]
# )
# history = not_pretrained.fit(
#     x_train, y_train, validation_data=(x_test, y_test), epochs=5, batch_size=64
# )


# repeat the same, but first pretraining at least one
base2 = CCLMModelBase(preprocessor=prep)

pretrainer_c = MaskedLanguagePretrainer(
    base=base2,
    downsample_factor=1,
    n_strided_convs=4,
    learning_rate=0.001,
    stride_len=1,
)

pretrainer_d = MaskedLanguagePretrainer(
    base=base2,
    downsample_factor=16,
    n_strided_convs=4,
)


pretrainer_c.fit(dataset_train[:1000], epochs=100, print_interval=20)

print(pretrainer_c.model.summary())
print(pretrainer_c.base.embedder.summary())
# import sys

# sys.exit(2)

# pretrainer_c.freeze()
# base2.freeze_embedder()


composed2 = ComposedModel(base2, [pretrainer_c.model, pretrainer_d.model])

# put a classification head on it
gmp = tf.keras.layers.GlobalMaxPool1D()
d = tf.keras.layers.Dense(4)
out = tf.keras.layers.Activation("softmax", dtype="float32")
pretrained = tf.keras.Model(composed2.model.input, out(d(gmp(composed2.model.output))))
pretrained.compile(
    tf.keras.optimizers.Adam(0.0005), "categorical_crossentropy", metrics=["accuracy"]
)
history_pretrained = pretrained.fit(
    x_train, y_train, validation_data=(x_test, y_test), epochs=1, batch_size=64
)


pretrainer_c.unfreeze()
base2.unfreeze_embedder()

pretrained.compile(
    tf.keras.optimizers.Adam(0.0005), "categorical_crossentropy", metrics=["accuracy"]
)
history_pretrained = pretrained.fit(
    x_train, y_train, validation_data=(x_test, y_test), epochs=15, batch_size=64
)
