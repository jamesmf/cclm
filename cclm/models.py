import numpy as np
import json
import os
from .preprocessing import MLMPreprocessor
import keras
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, SeparableConv1D, Conv1D, MaxPooling1D, Lambda, Multiply
from keras.layers import (
    Dropout,
    Dense,
    Embedding,
    Dot,
    Concatenate,
    Reshape,
    Add,
    Average,
)
from keras.layers import (
    GlobalMaxPooling1D,
    GlobalMaxPooling2D,
    Permute,
    RepeatVector,
    Flatten,
)
from keras.layers import BatchNormalization
from keras.layers.wrappers import TimeDistributed
from keras.regularizers import l1_l2
from keras.optimizers import Adam, RMSprop, SGD
from keras.models import Model, load_model
from keras.initializers import TruncatedNormal
from keras.constraints import MinMaxNorm
import keras.backend as K
import tensorflow as tf


def clipped_bce(y_true, y_pred):
    y_pred = K.clip(y_pred, 0.01, 0.9)
    #     thresh = 0.2
    bce = K.binary_crossentropy(y_true, y_pred)
    #     m = cce - thresh
    m = K.clip(bce, 0.01, 3.0)
    return K.mean(m, axis=-1)


def zero_out_zero_embedding(model):
    emb_layers = [l for l in model.layers if l.name.find("embedding") > -1]
    for layer in model.layers:
        if isinstance(layer, Embedding):
            print(f"zeroing out {layer.name}")
            w = layer.get_weights()
            w[0][0] = np.zeros_like(w[0][0])
            layer.set_weights(w)


def get_character_embedder(max_len, char_emb_size, n_chars, filters, prefix):
    emb_init = TruncatedNormal(mean=0.0001, stddev=0.2, seed=None)
    con = MinMaxNorm(0.05, 1.0)
    reg = l1_l2(0.0000001, 0.0000001)
    inp = Input((max_len,), name=f"{prefix}_inp")
    char_emb = Embedding(
        n_chars,
        char_emb_size,
        embeddings_constraint=con,
        embeddings_initializer=emb_init,
        name=f"{prefix}_embedding",
    )
    char_conv_1 = Conv1D(
        filters,
        3,
        activation="relu",
        name=f"{prefix}_c1",
        padding="same",
        kernel_constraint=con,
        bias_constraint=con,
    )
    char_conv_2 = Conv1D(
        filters,
        3,
        activation="relu",
        name=f"{prefix}_c2",
        padding="same",
        kernel_constraint=con,
        bias_constraint=con,
    )
    char_conv_3 = Conv1D(
        filters,
        3,
        activation="relu",
        name=f"{prefix}_c3",
        dilation_rate=2,
        padding="same",
        kernel_constraint=con,
        bias_constraint=con,
    )
    char_conv_4 = Conv1D(
        filters,
        3,
        activation="tanh",
        name=f"{prefix}_c4",
        dilation_rate=3,
        padding="same",
        kernel_constraint=con,
        bias_constraint=con,
    )
    char_conv_5 = Conv1D(
        4 * filters,
        3,
        activation="tanh",
        name=f"{prefix}_c5",
        dilation_rate=3,
        padding="same",
        kernel_constraint=con,
        bias_constraint=con,
    )
    x = char_emb(inp)
    cc1 = char_conv_1(x)
    x = Dropout(0.25)(cc1)
    x = char_conv_2(x)
    x = char_conv_3(x)
    r1 = Add(name=f"{prefix}_res_conn")([cc1, x])
    x = Dropout(0.25)(x)
    x = char_conv_4(x)
    r2 = Add(name=f"{prefix}_res_conn_2")([r1, x])
    x = char_conv_5(r2)
    return inp, x


class CCLMModelBase:
    def __init__(
        self,
        load_from=None,
        preprocessor=None,
        char_emb_size=32,
        n_filters=256,
        prefix="cclm",
        pool=False,
    ):
        self.char_emb_size = char_emb_size
        self.n_filters = n_filters
        self.prefix = prefix
        self.preprocessor = preprocessor
        if load_from:
            self._load(load_from, pool=pool)
        else:
            emb_in, emb_out = get_character_embedder(
                self.preprocessor.max_example_len,
                char_emb_size,
                np.max(list(self.preprocessor.char_dict.values())),
                n_filters,
                prefix,
            )
            self.embedder = Model(emb_in, emb_out)

    def fit(self, data):
        """
        Fit a basic embedder using the data provided and the default cclm
        pretraining task.
        """
        # if there's no preprocessor object yet, create one and fit it
        if self.preprocessor is None:
            self.preprocessor = Preprocessor()
            self.preprocessor.fit(data)
        pass

    def save(self, path):
        """
        Use keras to save the model and its preprocessor
        """
        pass

    def freeze_embedder(self):
        for layer in self.embedder.layers:
            if hasattr(layer, "trainable"):
                layer.trainable = False

    def unfreeze_embedder(self):
        for layer in self.embedder.layers:
            if hasattr(layer, "trainable"):
                layer.trainable = True

    def _load(self, path, pool=False):
        """
        Load a model and its preprocessor
        """
        emb_in, emb_out = get_character_embedder(
            self.preprocessor.max_example_len,
            self.char_emb_size,
            np.max(list(self.preprocessor.char_dict.values())),
            self.n_filters,
            self.prefix,
        )
        if pool:
            emb_out = GlobalMaxPooling1D(name="context_gmp")(emb_out)
        self.embedder = Model(emb_in, emb_out)
        loaded_model = load_model(path, custom_objects={"clipped_bce": clipped_bce})
        loaded_model.compile("Adam", "binary_crossentropy")
        print([layer.name for layer in loaded_model.layers])
        for layer in loaded_model.layers[:-1]:
            try:
                old_weights = layer.get_weights()
                if len(old_weights) == 0:
                    continue
                print(f"initializing layer: {layer.name}")
                new_layer = self.embedder.get_layer(layer.name)
                new_layer.set_weights(old_weights)
            except ValueError:
                print(f"unable to transfer weights for {layer.name}")

    def pretrain_embedder(self, sents, outname):
        gen = self.preprocessor.examples_generator(sents)
        val = self.preprocessor.examples_generator(sents)
        printer_callback = PrinterCallback(val, self.preprocessor)
        cb = [ModelCheckpoint(outname, save_best_only=False), printer_callback]
        to_pretrain = self.embedder
        to_pretrain = GlobalMaxPooling1D()(to_pretrain.output)
        to_pretrain = Dense(1, activation="sigmoid")(to_pretrain)
        model = Model(self.embedder.input, to_pretrain)
        model.compile("sgd", clipped_bce)
        model.fit_generator(
            gen,
            steps_per_epoch=5000,
            epochs=1000,
            validation_data=val,
            callbacks=cb,
            validation_steps=50,
        )


def rev(prep, a):
    return "".join([prep.char_rev[int(c)] for c in a])


class PrinterCallback(keras.callbacks.Callback):
    def __init__(self, gen, prep):
        super().__init__()
        self.gen = gen
        self.prep = prep

    def on_epoch_end(self, epoch, logs):
        ex = next(self.gen)
        sourceex = ex[0][0]
        print("*" * 80)
        label = ex[1][j]
        posex = ex[0][j]
        print(
            rev(self.prep, posex),
            ":",
            self.model.predict(ex[0][j : j + 1]),
            " -> ",
            label,
        )
        print("*" * 80)
