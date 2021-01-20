import numpy as np
import json
import os
from .preprocessing import MLMPreprocessor
import tensorflow as tf

DEFAULT_EMB_DIM = 128


def zero_out_zero_embedding(model):
    emb_layers = [l for l in model.layers if l.name.find("embedding") > -1]
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Embedding):
            print(f"zeroing out {layer.name}")
            w = layer.get_weights()
            w[0][0] = np.zeros_like(w[0][0])
            layer.set_weights(w)


def get_character_embedder(max_len, char_emb_size, n_chars, filters, prefix):
    inp = tf.keras.layers.Input((max_len,), name=f"{prefix}_inp")
    char_emb = tf.keras.layers.Embedding(
        n_chars,
        char_emb_size,
        name=f"{prefix}_embedding",
    )
    char_conv_1 = tf.keras.layers.Conv1D(
        filters,
        3,
        activation="relu",
        name=f"{prefix}_c1",
        padding="same",
    )
    char_conv_2 = tf.keras.layers.Conv1D(
        filters,
        3,
        activation="relu",
        name=f"{prefix}_c2",
        padding="same",
    )
    char_conv_3 = tf.keras.layers.Conv1D(
        filters,
        3,
        activation="relu",
        name=f"{prefix}_c3",
        dilation_rate=2,
        padding="same",
    )
    char_conv_4 = tf.keras.layers.Conv1D(
        filters,
        3,
        activation="tanh",
        name=f"{prefix}_c4",
        dilation_rate=3,
        padding="same",
    )
    char_conv_5 = tf.keras.layers.Conv1D(
        4 * filters,
        3,
        activation="tanh",
        name=f"{prefix}_c5",
        dilation_rate=3,
        padding="same",
    )
    x = char_emb(inp)
    cc1 = char_conv_1(x)
    x = tf.keras.layers.Dropout(0.25)(cc1)
    x = char_conv_2(x)
    x = char_conv_3(x)
    r1 = tf.keras.layers.Add(name=f"{prefix}_res_conn")([cc1, x])
    x = tf.keras.layers.Dropout(0.25)(x)
    x = char_conv_4(x)
    r2 = tf.keras.layers.Add(name=f"{prefix}_res_conn_2")([r1, x])
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
                np.max(list(self.preprocessor.char_dict.values())) + 1,
                n_filters,
                prefix,
            )
            self.embedder = tf.keras.Model(emb_in, emb_out)

    def save(self, path):
        """
        Use keras to save the model and its preprocessor
        """
        self.embedder.save(path)

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
            emb_out = tf.keras.layers.GlobalMaxPooling1D(name="context_gmp")(emb_out)
        self.embedder = tf.keras.Model(emb_in, emb_out)
        loaded_model = load_model(path)
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
        to_pretrain = tf.keras.layers.GlobalMaxPooling1D()(to_pretrain.output)
        to_pretrain = tf.keras.layers.Dense(1, activation="sigmoid")(to_pretrain)
        model = tf.keras.Model(self.embedder.input, to_pretrain)
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


class PrinterCallback(tf.keras.callbacks.Callback):
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


class MultiHeadSelfAttention(tf.keras.layers.Layer):
    """
    From keras examples
    https://keras.io/examples/nlp/text_classification_with_transformer/
    """

    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = tf.keras.layers.Dense(embed_dim)
        self.key_dense = tf.keras.layers.Dense(embed_dim)
        self.value_dense = tf.keras.layers.Dense(embed_dim)
        self.combine_heads = tf.keras.layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(
            query, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(
            key, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(
            value, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(
            concat_attention
        )  # (batch_size, seq_len, embed_dim)
        return output


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        embed_dim: int = DEFAULT_EMB_DIM,
        num_heads: int = 8,
        ff_dim: int = 32,
        rate=0.1,
    ):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(ff_dim, activation="relu"),
                tf.keras.layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TestBlock(tf.keras.layers.Layer):
    def __init__(self):
        super(TestBlock, self).__init__()
        self.d = tf.keras.layers.Dense(10)

    def call(self, inputs, training):
        return self.d(inputs)
