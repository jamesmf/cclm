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

    def fit(
        self,
        data: List[str],
        epochs: int = 1,
        batch_size: int = 32,
        print_interval: int = 100,
        evaluate_interval: int = 1000,
    ):
        """
        This function should perform some form of optimization over a provided dataset.
        Even though the task may involve many inputs and outputs, the result should
        be that the .model gets fit such that it accets input from a `base` and
        produces output that matches the shape of the input.
        """
        raise NotImplementedError

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


class MaskedLanguagePretrainer(Pretrainer):
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
        vocab_size: int = 10000,
        **kwargs,
    ):
        """
        Pretrain a model by doing masked language modeling on a corpus.

        Model that is trained accepts the base input with shape (batch_size, example_len, base_embedding_dim)
        and performs convolutions on the input. The input can be downsampled using strided conv layers,
        making the Transformer layer less expensive. The kernel_size of the conv layers
        also match the stride_len for simplicity
        """
        self.preprocessor = kwargs.get("preprocessor", MLMPreprocessor())
        base = kwargs.get("base")
        if base:
            self.preprocessor = base.preprocessor
        self.n_strided_convs = n_strided_convs
        self.downsample_factor = downsample_factor
        # calculate stride length to achieve the right downsampling
        assert (
            stride_len ** n_strided_convs == downsample_factor
        ), "stride_len^n_strided_convs must equal downsample_factor"
        self.stride_len = int(stride_len)
        self.n_conv_filters = n_conv_filters
        self.pool = tf.keras.layers.GlobalMaxPool1D(dtype="float32")
        self.optimizer = tf.optimizers.SGD(learning_rate)

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
        super().__init__(*args, **kwargs)

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
                # TransformerBlock(embed_dim=self.n_conv_filters),
                # tf.keras.layers.UpSampling1D(size=self.downsample_factor),
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
    ) -> Tuple[List[str], List[Tuple[int, int]], List[int]]:
        """
        Transform input strings into correct-length substrings and pick tokens to mask
        """
        batch_inputs: List[str] = []
        batch_outputs: List[int] = []
        batch_spans: List[Tuple[int, int]] = []
        tokenizer = self.preprocessor.tokenizer
        for example in input_strs:

            # subset to a substring of the correct len
            # encoded = self.get_substr(encoded)
            example = self.get_substr(example)
            encoded = tokenizer.encode(example)

            # otherwise, mask a token and predict it
            possible_masked_tokens = encoded.ids
            masked_token_index = np.random.randint(
                0, len(possible_masked_tokens)
            )  # +1 for [CLS] token we skipped
            masked_token = encoded.ids[masked_token_index]
            start, end = encoded.token_to_chars(masked_token_index)
            masked_token_len = end - start
            inp = (
                example[:start]
                + "?" * masked_token_len
                + example[start + masked_token_len :]
            )
            batch_inputs.append(inp)
            batch_spans.append((start, end))
            batch_outputs.append(masked_token)
        return batch_inputs, batch_spans, batch_outputs

    def fit(
        self,
        data: List[str],
        epochs: int = 1,
        batch_size: int = 32,
        print_interval: int = 100,
        evaluate_interval: int = 1000,
    ):
        """
        Iterate over a corpus of strings, using the preprocessor's tokenizer to mask
        some tokens, and predicting the masked tokens.
        """
        for ep in range(epochs):
            batch_strs: List[str] = []
            losses = []
            n_batches_completed = 0
            for n, example in enumerate(tqdm(data)):
                example = example.strip()

                if not self.can_learn_from(example):
                    continue
                batch_strs.append(example)

                # if we've accumulated enough valid examples for a batch or the epoch is over
                if len(batch_strs) == batch_size or (
                    n + 1 == len(data) and len(batch_strs) > 0
                ):
                    batch_inputs, batch_spans, batch_outputs = self.batch_from_strs(
                        batch_strs
                    )

                    x = self.get_batch(batch_inputs)
                    y = np.array(batch_outputs)
                    if (n_batches_completed + 1) % evaluate_interval == 0:
                        evaluation = self.evaluate(x, y, batch_inputs)
                        print(evaluation)
                    loss = self.train_batch(x, y, batch_spans)
                    n_batches_completed += 1
                    losses.append(loss)
                    if n_batches_completed % print_interval == 0:
                        tqdm.write(
                            f"Mean loss after {n_batches_completed} batches: {np.mean(losses)}"
                        )

                    # reset batch
                    batch_inputs, batch_outputs, batch_spans, batch_strs = (
                        [],
                        [],
                        [],
                        [],
                    )

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

        # identify which tokens could be a start token
        possible_start_tokens = [i for i in inp.offsets if i[1] + length <= max_ind]

        return inp

    def get_batch(self, input_strings: List[str]) -> np.ndarray:
        """
        Return vector encoding of a batch of strings using the preprocessor
        """
        max_len = np.max(list(map(len, input_strings)))
        out = np.zeros((len(input_strings), max_len))
        for n, s in enumerate(input_strings):
            out[n] = self.preprocessor.string_to_array(s, max_len)
        return out

    def train_batch(
        self, x: np.ndarray, y: np.ndarray, batch_spans: List[Tuple[int, int]]
    ):
        with tf.GradientTape() as g:
            rep = self.model(self.base.embedder(x, training=True), training=True)
            mask = np.zeros(rep.shape[:-1])
            for n, span in enumerate(batch_spans):
                mask[n][span[0] : span[1] + 1] = 1
            masked_word_rep = self.pool(rep * mask.reshape(*mask.shape, 1))
            loss = self.get_loss(masked_word_rep, y)
            to_diff = (
                self.model.trainable_weights + self.base.embedder.trainable_weights
                if self.train_base
                else [] + [self.output_weights, self.output_biases]
            )
            gradients = g.gradient(loss, to_diff)
            self.optimizer.apply_gradients(zip(gradients, to_diff))
        return loss

    def get_loss(self, x_inp: np.ndarray, y: np.ndarray):
        with tf.device("/GPU:0"):
            # Compute the average loss for the batch.
            y_true = tf.cast(tf.reshape(y, [-1, 1]), tf.int64)
            sampled = tf.random.learned_unigram_candidate_sampler(
                y_true,
                1,
                64,
                True,
                self.preprocessor.tokenizer.get_vocab_size() + 1,
                name="learned_unigram_dist",
            )
            loss = tf.reduce_mean(
                tf.nn.sampled_softmax_loss(
                    weights=self.output_weights,
                    biases=self.output_biases,
                    labels=y_true,
                    inputs=x_inp,
                    num_sampled=64,
                    sampled_values=sampled,
                    num_classes=self.preprocessor.tokenizer.get_vocab_size() + 1,
                    num_true=1,
                )
            )
            return loss

    def evaluate(
        self,
        x: np.ndarray,
        y: np.ndarray,
        batch_inputs: List[str],
        n_examples: int = 5,
        n_nearest: int = 5,
    ):
        training = True  # if this improves evaluation, there's a bug
        print_str = ""
        # Compute the cosine similarity between input data embedding and every embedding vector
        rep = self.pool(
            self.model(self.base.embedder(x, training=training), training=training),
            training=training,
        )
        x_embed = tf.cast(rep, tf.float32)
        x_embed_norm = x_embed / tf.sqrt(tf.reduce_sum(tf.square(x_embed)))
        embedding_norm = self.output_weights / tf.sqrt(
            tf.reduce_sum(tf.square(self.output_weights), 1, keepdims=True), tf.float32
        )
        cosine_sim = tf.matmul(x_embed_norm, embedding_norm, transpose_b=True).numpy()
        for i in range(n_examples):
            print_str += f"Example:\n\n{batch_inputs[i]}\n"
            nearest = (-cosine_sim[i, :]).argsort()[:n_nearest]
            print_str += f"ground truth: {self.preprocessor.tokenizer.id_to_token(int(y[i]))}\n\npredicted:\n"

            for k in range(n_nearest):
                print_str += "\n-" + self.preprocessor.tokenizer.id_to_token(nearest[k])
        return print_str
