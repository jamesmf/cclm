import json
import os
from typing import List, Tuple
import numpy as np
from tqdm import tqdm
from tokenizers import Encoding
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    Multiply,
    Dropout,
    Dense,
    Concatenate,
    GlobalMaxPooling1D,
    Permute,
)
import tensorflow.keras.backend as K
import tensorflow as tf
from .models import CCLMModelBase, TransformerBlock
from .preprocessing import MLMPreprocessor


class Pretrainer:
    """
    A pretrainer needs to accept a base, implement some core layers that it
    will fit (in addition to the base, optionally), and implement a top to the
    network that represents its specific task.

    The
    """

    def __init__(
        self,
        base=None,
        task_name="pretraining",
        load_from: str = None,
        base_args={},
        **kwargs,
    ):
        if load_from:
            self.model = tf.keras.models.load_model(load_from)
        else:
            self.model = self.get_model()
        if base is None:
            base = CCLMModelBase(**base_args)
        self.base = base
        self.task_name = task_name

    def get_model(self, *args, **kwargs):
        """
        This should return a Model that accepts input with the shape of a `base` and
        produces output of the same shape.
        """

    def compile(self, *args, **kwargs):
        self.model.compile(*args, **kwargs)

    def save_model(self, path: str):
        self.model.save(path)

    def freeze(self):
        """
        Make the layers in the model not trainable. Useful when first combining pretrained models
        with randomly initialized layers
        """
        for layer in self.model.layers:
            if hasattr(layer, "trainable"):
                layer.trainable = False

    def unfreeze(self):
        """
        Make the layers in the model not trainable.
        """
        for layer in self.model.layers:
            if hasattr(layer, "trainable"):
                layer.trainable = True


class MaskedLanguagePretrainer(tf.keras.Model, Pretrainer):
    def __init__(
        self,
        *args,
        n_conv_filters: int = 128,
        downsample_factor: int = 4,
        n_strided_convs: int = 2,
        stride_len: int = 2,
        mask_id: int = 1,
        learning_rate: float = 0.001,
        train_base: bool = True,
        training_pool_mode: str = "local",
        min_mask_len: int = 3,
        num_negatives=5,
        **kwargs,
    ):
        """
        Pretrain a model by doing masked language modeling on a corpus.

        Model that is trained accepts the base input with shape (batch_size, example_len, base_embedding_dim)
        and performs convolutions on the input. The input can be downsampled using strided conv layers,
        making the Transformer layer less expensive. The kernel_size of the conv layers
        also match the stride_len for simplicity
        """
        tf.keras.Model.__init__(self)
        self.preprocessor = kwargs.get("preprocessor", MLMPreprocessor())
        base = kwargs.get("base")
        if base:
            self.preprocessor = base.preprocessor
        self.n_strided_convs = n_strided_convs
        self.downsample_factor = downsample_factor
        self.training_pool_mode = training_pool_mode
        self.min_mask_len = min_mask_len
        self.mask_id = mask_id
        # calculate stride length to achieve the right downsampling
        assert (
            stride_len ** n_strided_convs == downsample_factor
        ), "stride_len^n_strided_convs must equal downsample_factor"
        self.stride_len = int(stride_len)
        self.n_conv_filters = n_conv_filters
        self.pool = tf.keras.layers.GlobalMaxPool1D(dtype="float32")
        self.optimizer = tf.optimizers.SGD(learning_rate)
        self.num_negatives = num_negatives

        self.train_base = train_base

        # initialize output weights for the sampled softmax or nce loss
        self.output_weights = tf.Variable(
            tf.random.normal(
                [self.preprocessor.tokenizer.get_vocab_size() + 1, self.n_conv_filters]
            )
        )
        self.output_biases = tf.Variable(
            tf.zeros([self.preprocessor.tokenizer.get_vocab_size() + 1])
        )
        # negative sampling head
        self.classification_head = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(128, activation="tanh"),
                tf.keras.layers.Dense(1, activation="sigmoid", dtype="float32"),
            ]
        )
        self.concat = tf.keras.layers.Concatenate()
        self.output_embedding = tf.keras.layers.Embedding(
            self.preprocessor.tokenizer.get_vocab_size() + 1, self.n_conv_filters
        )
        Pretrainer.__init__(self, *args, **kwargs)
        self.pretraining_model = self.get_pretraining_model()

    def get_model(
        self,
    ):
        """
        Until handled better, inputs need to be padded to a multiple of filter_stride_len*n_strided_convs.

        The model uses one or more strided Conv1D to reduce the input shape before passing it to
        one or more transformer blocks
        """
        # reduce the size, transformer, upsample

        layers = [
            Conv1D(
                self.n_conv_filters,
                self.stride_len,
                strides=self.stride_len,
                padding="same",
                activation="tanh",
            )
            for _ in range(self.n_strided_convs)
        ]
        model = tf.keras.Sequential(
            [
                Conv1D(
                    self.n_conv_filters,
                    self.stride_len,
                    padding="same",
                    activation="tanh",
                ),
                tf.keras.layers.Dropout(0.2),
                *layers,
                tf.keras.layers.Dropout(0.2),
                TransformerBlock(embed_dim=self.n_conv_filters),
                *[
                    tf.keras.layers.UpSampling1D(size=self.downsample_factor)
                    for _ in range(1)
                    if self.downsample_factor > 1
                ],
                Conv1D(
                    self.n_conv_filters,
                    self.stride_len,
                    padding="same",
                    activation="tanh",
                ),
            ]
        )

        return model

    def can_learn_from(self, example_str: str) -> bool:
        """
        Decide whether it's appropriate to learn from this example
        """
        # if it's an empty string, skip it
        if example_str == "":
            return False

        # if it's just a [CLS] a few tokens and [SEP], skip it
        if len(example_str) < 20:
            return False
        return True

    def batch_from_strs(
        self, input_strs: List[str]
    ) -> Tuple[List[str], List[Tuple[int, int, str]], List[int]]:
        """
        Transform input strings into correct-length substrings and pick tokens to mask
        """
        batch_inputs: List[str] = []
        batch_outputs: List[int] = []
        batch_spans: List[Tuple[int, int, str]] = []
        tokenizer = self.preprocessor.tokenizer
        for example in input_strs:

            # subset to a substring of the correct len
            # encoded = self.get_substr(encoded)
            example = self.get_substr(example)
            encoded = tokenizer.encode(example)

            # get all tokens that are long enough to be masked
            possible_masked_tokens = [
                _id
                for n, _id in enumerate(encoded.ids)
                if len(encoded.tokens[n]) >= self.min_mask_len
            ]
            possible_encoding_indexes = [
                n
                for n, token in enumerate(encoded.tokens)
                if len(token) >= self.min_mask_len
            ]
            # if none, pick a shorter token
            if len(possible_masked_tokens) == 0:
                possible_masked_tokens = encoded.ids
                possible_encoding_indexes = [n for n in range(0, len(encoded.ids))]
            # sample a value to index into possible_masked_tokens
            masked_token_index = np.random.randint(0, len(possible_masked_tokens))
            # look up the value sampled
            masked_token_id = possible_masked_tokens[masked_token_index]
            start, end = encoded.token_to_chars(
                possible_encoding_indexes[masked_token_index]
            )
            masked_token_len = end - start
            inp = (
                example[:start]
                + "?" * masked_token_len
                + example[start + masked_token_len :]
            )
            batch_inputs.append(inp)
            batch_spans.append((start, end, example[start : start + masked_token_len]))
            batch_outputs.append(masked_token_id)
        return batch_inputs, batch_spans, batch_outputs

    def generator(self, data: List[str], batch_size: int):
        """
        Generator for training purposes
        """
        while True:
            batch_inputs, batch_outputs, batch_spans, batch_strs = (
                [],
                [],
                [],
                [],
            )
            for n, example in enumerate(data):
                example = example.strip()

                if not self.can_learn_from(example):
                    continue
                batch_strs.append(example)
                if len(batch_strs) == batch_size or (
                    n + 1 == len(data) and len(batch_strs) > 0
                ):
                    batch_inputs, batch_spans, batch_outputs = self.batch_from_strs(
                        batch_strs
                    )

                    x = self.get_batch(batch_inputs, batch_spans=batch_spans)
                    x = tf.concat([x, *[x for _ in range(self.num_negatives)]], axis=0)

                    y = np.array(batch_outputs)
                    y_sample, token_sample, mask = self.sample_from_positive(
                        y, x, batch_spans
                    )
                    yield [x, token_sample, mask], y_sample
                    batch_inputs, batch_outputs, batch_spans, batch_strs = (
                        [],
                        [],
                        [],
                        [],
                    )

    def train_step(self, data):
        """
        Iterate over a corpus of strings, using the preprocessor's tokenizer to mask
        some tokens, and predicting the masked tokens.
        """
        x, y_true = data
        x_char, x_token, mask = x
        loss, preds = self.train_batch(x_char, x_token, y_true, mask)
        self.compiled_metrics.update_state(y_true, preds)
        return {m.name: m.result() for m in self.metrics}

    def get_substr(self, inp: str) -> str:
        """
        Return a substring that is an appropriate length
        """
        max_len = self.preprocessor.max_example_len
        inp_len = len(inp)
        if inp_len <= max_len:
            return inp
        start = np.random.randint(0, inp_len - max_len)
        return inp[start : start + max_len]

    def account_for_downsample(self, inp_len: int) -> int:
        """
        Since the transformer model might downsample, we might need to do extra padding
        """
        return 32
        return inp_len + (inp_len % self.downsample_factor)

    def get_batch(
        self, input_strings: List[str], batch_spans: List[Tuple[int, int, str]] = None
    ) -> np.ndarray:
        """
        Return vector encoding of a batch of strings using the preprocessor
        """
        max_len = self.account_for_downsample(np.max(list(map(len, input_strings))))
        out = np.zeros((len(input_strings), max_len))
        for n, s in enumerate(input_strings):
            out[n] = self.preprocessor.string_to_array(s, max_len)
            if batch_spans is not None:
                start, end, _n = batch_spans[n]
                out[n][start:end] = self.mask_id
        return out

    def train_batch(
        self, x_char: np.ndarray, x_token: np.ndarray, y: np.ndarray, mask: np.ndarray
    ):
        with tf.GradientTape() as g:
            # char_rep = self.get_embedding_for_pretraining(
            #     x_char, mask=mask, training=True
            # )
            # token_emb = self.output_embedding(x_token)

            # loss, predictions = self.get_loss(char_rep, token_emb, y)
            predictions = self([x_char, x_token, mask], training=True)
            loss = self.compiled_loss(y, predictions)
            to_diff = (
                self.model.trainable_weights + self.base.embedder.trainable_weights
                if self.train_base
                else []
                + [
                    self.output_embedding.trainable_weights,
                    self.classification_head.trainable_weights,
                ]
            )
            gradients = g.gradient(loss, to_diff)
            self.optimizer.apply_gradients(zip(gradients, to_diff))

        return loss, predictions

    def get_embedding_for_pretraining(
        self, x: np.ndarray, mask, training=True
    ) -> np.ndarray:
        """
        Embed the text and possibly pool the whole thing or pool just the masked token
        """
        rep = self.model(self.base.embedder(x, training=training), training=training)
        # either pool the whole representation
        if self.training_pool_mode == "global":
            rep = self.pool(rep)
        else:
            rep = self.pool(self.mask_mul([rep, mask]))
        return rep

    def sample_from_positive(self, y, x, batch_spans):
        with tf.device("/GPU:0"):
            # Compute the average loss for the batch.
            y_true = tf.cast(tf.reshape(y, [-1, 1]), tf.int64)
            # sampled = tf.random.learned_unigram_candidate_sampler(
            (
                neg_sample,
                true_expected,
                neg_expected,
            ) = tf.random.log_uniform_candidate_sampler(
                y_true,
                1,
                self.num_negatives * y_true.shape[0],
                True,
                self.preprocessor.tokenizer.get_vocab_size() + 1,
                name="sampler_dist",
            )
            # true samples get a 1, negative samples get a 0
            y_sample = tf.concat(
                (tf.ones((y_true.shape[0], 1)), tf.zeros((neg_sample.shape[0], 1))),
                axis=0,
            )
            token_sample = tf.reshape(
                tf.concat((y_true, tf.reshape(neg_sample, (-1, 1))), axis=0), (-1,)
            )
            mask = np.zeros_like(x)
            ind = 0
            for span in batch_spans:
                mask[ind][span[0] : span[1]] = 1.0
                ind += 1

            for _ in range(self.num_negatives):
                for span in batch_spans:
                    mask[ind][span[0] : span[1]] = 1
                    ind += 1
            mask = mask.reshape((*mask.shape[:2], 1))
            # mask = tf.cast(mask, tf.float16)
            return y_sample, token_sample, mask

    def get_loss(
        self,
        rep_char: tf.Tensor,
        rep_token: tf.Tensor,
        y_sample: np.ndarray,
    ):
        with tf.device("/GPU:0"):

            cat = self.concat([rep_char, rep_token])
            predictions = self.classification_head(cat)
            loss = self.compiled_loss(y_sample, predictions)
            return loss, predictions

    def get_pretraining_model(self):
        """
        Return a Model for training using negative sampling
        """
        inp_char = tf.keras.layers.Input((None,))
        inp_token = tf.keras.layers.Input((1,))
        inp_mask = tf.keras.layers.Input((None,))

        rep = self.base.embedder(inp_char)
        rep = self.model(rep)
        mul = self.pool(tf.keras.layers.Multiply()([rep, inp_mask]))

        emb_token = self.pool(self.output_embedding(inp_token))
        cat = self.concat([emb_token, mul])
        out = self.classification_head(cat)
        return tf.keras.Model([inp_char, inp_token, inp_mask], out)

    def call(self, x, training=False):
        return self.pretraining_model(x, training=training)

    def evaluate(
        self,
        x: np.ndarray,
        y: np.ndarray,
        batch_inputs: List[str],
        batch_spans: List[Tuple[int, int, str]],
        n_examples: int = 5,
        n_nearest: int = 5,
    ):
        return ""
        training = True  # if this improves evaluation, there's a bug
        print_str = ""
        # Compute the cosine similarity between input data embedding and every embedding vector
        rep = self.get_embedding_for_pretraining(
            x, batch_spans=batch_spans, training=True
        )

        logits = tf.matmul(rep, tf.transpose(self.output_weights))
        logits = tf.nn.bias_add(logits, self.output_biases).numpy()

        for i in range(n_examples):
            print_str += f"\nExample {i}:\n\n{batch_inputs[i]}\n"
            nearest = (-logits[i, :]).argsort()[:n_nearest]
            print_str += f"ground truth: {self.preprocessor.tokenizer.id_to_token(int(y[i]))}\n\npredicted:\n"

            for k in range(n_nearest):
                print_str += "\n-" + self.preprocessor.tokenizer.id_to_token(nearest[k])
        return print_str
