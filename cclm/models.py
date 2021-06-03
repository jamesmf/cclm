from typing import Any, Generator, List, Dict, Optional
import numpy as np
import json
import os
from .preprocessing import Preprocessor
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision

policy = mixed_precision.Policy("mixed_float16")
mixed_precision.set_policy(policy)


DEFAULT_EMB_DIM = 128


class GlobalBroadcaster1D(tf.keras.layers.Layer):
    def __init__(self, dim: int):
        super().__init__()
        self.pooler = tf.keras.layers.GlobalMaxPool1D()
        self.reshaper = tf.keras.layers.Reshape((1, -1))
        self.dense = tf.keras.layers.Dense(dim, activation="relu")
        self.cat = tf.keras.layers.Concatenate()

    def call(self, inputs, training):
        # globally pool and broadcast back
        side = self.reshaper(self.pooler(inputs))
        side = tf.repeat(self.dense(side), tf.shape(inputs)[1], axis=1)
        return self.cat([inputs, side])


def zero_out_zero_embedding(model: tf.keras.Model):
    emb_layers = [l for l in model.layers if l.name.find("embedding") > -1]
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Embedding):
            print(f"zeroing out {layer.name}")
            w = layer.get_weights()
            w[0][0] = np.zeros_like(w[0][0])
            layer.set_weights(w)


class Embedder:
    persisted_attrs: List[str] = [
        "max_len",
        "char_emb_size",
        "n_blocks",
        "n_chars",
        "n_filters",
        "global_filters",
        "filter_len",
    ]
    type_desc = "embedder"
    config_name = "config.json"

    def __init__(
        self,
        max_len: int = 1024,
        n_chars: int = 1000,
        char_emb_size: int = 32,
        n_blocks: int = 5,
        n_filters: int = 512,
        global_filters: int = 128,
        filter_len: int = 3,
        load_from: Optional[str] = None,
    ) -> None:
        self.max_len = max_len
        self.char_emb_size = char_emb_size
        self.n_blocks = n_blocks
        self.n_chars = n_chars
        self.n_filters = n_filters
        self.global_filters = global_filters
        self.filter_len = filter_len

        # if load_from left out the type_desc suffix add it
        if load_from and os.path.split(load_from)[-1] != self.type_desc:
            load_from = os.path.join(load_from, self.type_desc)

        if load_from:
            with open(os.path.join(load_from, self.config_name), "r") as f:
                config = json.load(f)
                self.from_config(config)

        inp_layer = tf.keras.layers.Input((self.max_len))
        self.model = self.get_character_embedder(inp_layer, "cclm_embedder")
        if load_from:
            self.model.load_weights(load_from)

    def from_config(self, config: Dict[str, Any]) -> None:
        for key, value in config.items():
            if isinstance(key, str):
                setattr(self, key, value)
            else:
                print(f"could not load {key} with value {value} from config")

    def save_config(self, dir: str):
        config = {key: getattr(self, key) for key in self.persisted_attrs}
        with open(os.path.join(dir, self.config_name), "w") as f:
            json.dump(config, f, indent=2)

    def save_model(self, dir: str):
        self.model.save_weights(dir)

    def save(self, dir: str):
        new_dir = os.path.join(dir, self.type_desc)
        os.makedirs(new_dir, exist_ok=True)
        self.save_config(new_dir)
        self.save_model(new_dir)

    def get_block(
        self,
        x: tf.Tensor,
        prefix: str,
        suffix: str,
    ):
        """
        Add a block of layers on top of input tensor
        """
        char_conv_1 = tf.keras.layers.Conv1D(
            self.n_filters,
            self.filter_len,
            activation=tf.keras.layers.LeakyReLU(alpha=0.1),
            name=f"{prefix}_conv_{suffix}_1",
            padding="same",
        )
        char_conv_2 = tf.keras.layers.Conv1D(
            self.n_filters,
            self.filter_len,
            activation=tf.keras.layers.LeakyReLU(alpha=0.1),
            name=f"{prefix}_conv_{suffix}_2",
            padding="same",
        )
        char_conv_3 = tf.keras.layers.Conv1D(
            self.n_filters,
            self.filter_len,
            activation=tf.keras.layers.LeakyReLU(alpha=0.1),
            name=f"{prefix}_conv_{suffix}_3",
            padding="same",
        )
        broadcaster_1 = GlobalBroadcaster1D(self.global_filters)
        cc1 = char_conv_1(x)
        x = tf.keras.layers.Dropout(0.25)(cc1)
        x = char_conv_2(x)
        x = broadcaster_1(x)
        x = char_conv_3(x)
        return tf.keras.layers.Add(name=f"{prefix}_res_{suffix}")([cc1, x])

    def get_character_embedder(
        self,
        inp_layer: tf.keras.layers.Input,
        prefix: str,
    ) -> tf.keras.Model:
        """
        Return a basic model that embeds character-level input and passes it through conv
        layers that don't change its length.

        Also aggregated sequence-level info by pooling then multiplying to make global information
        available in lower levels

        TODO: make this into a class so it's more obvious how to write your own
        """
        char_emb = TokenAndPositionEmbedding(
            self.max_len,
            self.n_chars,
            self.char_emb_size,
        )
        # initial input and conv
        x = char_emb(inp_layer)

        for n in range(self.n_blocks):
            x = self.get_block(
                x,
                prefix,
                str(n),
            )
        return tf.keras.Model(inp_layer, x)


def rev(prep: Preprocessor, a: str):
    return "".join([prep.char_rev[int(c)] for c in a])


class PrinterCallback(tf.keras.callbacks.Callback):
    def __init__(self, gen: Generator, prep: Preprocessor):
        super().__init__()
        self.gen = gen
        self.prep = prep

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]):
        j = 0
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


class PositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, maxlen: int, embed_dim: int):
        super(PositionEmbedding, self).__init__()
        self.pos_emb = tf.keras.layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-2]
        positions_in = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions_in)
        out = x + positions
        return out


class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = tf.keras.layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.pos_emb = PositionEmbedding(maxlen, embed_dim)

    def call(self, x):
        x = self.token_emb(x)
        return self.pos_emb(x)


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
        dim_key = tf.cast(tf.shape(key)[-1], tf.float16)
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


class ComposedModel(tf.keras.layers.Layer):
    def __init__(
        self,
        embedder: Embedder,
        components: List[tf.keras.Model],
        name: str = "composed_model",
    ):
        """
        Holds a set of pretrained models that share a base. When called, it will
        run the input through the base embedder, call each model on that embedding,
        and concatenate the output along the last dimension
        """
        super().__init__()
        self.models = components
        self.embedder = embedder
        self.model_name = name
        self.model = self.build_model()

    def build_model(self):
        emb = self.embedder.model

        towers = []
        for model_component in self.components:
            x = model_component(emb.output)
            towers.append(x)
        cat = tf.keras.layers.Concatenate()(towers)
        return tf.keras.Model(emb.input, cat, name=self.model_name)

    def call(self, inputs, training):
        return self.model(inputs, training=training)
