from .core import Pretrainer
import tensorflow as tf
import numpy as np
from datasets.arrow_dataset import Dataset
from typing import List, Union


class BasePretrainer(Pretrainer):
    """
    Pretrain a CCLMBase embedder with a masked-character-like model.

    This task is to take an input, apply noisy transformations
    (including the mask), and predict the original input
    """

    def __init__(
        self,
        base=None,
        task_name="pretraining",
        load_from: str = None,
        base_args={},
        **kwargs,
    ):
        self.augmentor = kwargs.pop("augmentor", None)
        self.dropout_rate = kwargs.pop("dropout_rate", 0.1)
        super().__init__(
            base=base,
            task_name=task_name,
            load_from=load_from,
            base_args=base_args,
            **kwargs,
        )

    def get_model(self):
        """
        SDO1D + LSTM
        """
        n_characters = self.base.n_chars + 1
        inp = self.base.embedder.input
        emb_out = self.base.embedder.output
        drop = tf.keras.layers.Dropout(0.2)
        lstm = tf.keras.layers.LSTM(128, return_sequences=True)
        dense = tf.keras.layers.Dense(
            n_characters + 1, activation="softmax", dtype="float32"
        )
        return tf.keras.Model(inp, dense(lstm(drop(emb_out))))

    # def fit(self, data, batch_size=32):
    #     """
    #     self.model.fit(gen)
    #     """

    def generator(self, data: Union[Dataset, List[str]], batch_size: int = 32):
        """
        Return a generator over the dataset that vectorizes the text and applies
        the noisy transformation to the input.
        """
        is_dataset = isinstance(data, Dataset)
        x, y = [], []
        while True:
            for n, example in enumerate(data):
                if is_dataset:
                    example = example["text"]
                if len(example) < self.preprocessor.max_example_len:
                    continue
                substr = self.get_substr(example)
                y_str = self.preprocessor.string_to_array(
                    substr, self.preprocessor.max_example_len
                )
                if self.augmentor:
                    substr = self.augmentor.transform(substr)
                x_str = self.preprocessor.string_to_array(
                    substr, self.preprocessor.max_example_len
                )
                x_str = np.where(
                    np.random.rand(*x_str.shape) < self.dropout_rate,
                    self.preprocessor.mask_token_ind,
                    x_str,
                )
                x.append(x_str)
                y.append(y_str)
                if len(x) == batch_size or n + 1 == len(data):
                    yield np.array(x), np.array(y)
                    x, y = [], []

    def evaluate_prediction(self, x: np.ndarray, pred: np.ndarray):
        """
        evaluate the predicted string compared to the input
        """
        out_str = ""
        rev = self.preprocessor.char_rev
        maxes = np.argmax(pred, axis=2)
        for n in range(len(x)):
            out_str += f"Example {n}:\n"
            out_str += "".join(rev[i] for i in x[n]) + "\n"
            out_str += "".join(rev[i] for i in maxes[n]) + "\n\n"
        return out_str


class BasePretrainerEvaluationCallback(tf.keras.callbacks.Callback):
    def __init__(self, data: List[str], pretrainer: BasePretrainer, log_every: int = 5):
        super().__init__()
        self.pretrainer = pretrainer
        self.log_every = log_every
        self.gen = pretrainer.generator(data)

    def on_epoch_end(self, epoch, logs):
        if epoch % self.log_every == 0:
            x, y = next(self.gen)
            log_str = self.pretrainer.evaluate_prediction(x, self.model(x))
            print(log_str)
        return
