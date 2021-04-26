from .core import Pretrainer
import tensorflow as tf
import numpy as np


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
        cd = self.base.preprocessor.char_dict
        n_characters = max(list(cd.values())) + 1
        inp = self.base.embedder.input
        emb_out = self.base.embedder.output
        drop = tf.keras.layers.Dropout1D(0.1)
        lstm = tf.keras.layers.LSTM(128, return_sequences=True)
        dense = tf.keras.layers.Dense(
            n_characters, activation="softmax", dtype="float32"
        )
        return tf.keras.Model(inp, dense(lstm(drop(emb_out))))

    # def fit(self, data, batch_size=32):
    #     """
    #     self.model.fit(gen)
    #     """

    def generator(self, data, batch_size=32):
        """
        Return a generator over the dataset that vectorizes the text and applies
        the noisy transformation to the input.
        """
        while True:
            x, y = [], []
            for example in data:
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
                x.append(x_str)
                y.append(y_str)
                if len(x) == batch_size:
                    yield np.array(x), np.array(y)
                    x, y = [], []