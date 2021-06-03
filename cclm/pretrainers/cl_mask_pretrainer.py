import os
import json
from cclm.preprocessing import Preprocessor
from tensorflow.python.autograph.pyct import transformer
from .core import Pretrainer
from ..models import Embedder, TransformerBlock, PositionEmbedding
from ..augmentation import Augmentor
import tensorflow as tf
import numpy as np
from datasets.arrow_dataset import Dataset
from typing import List, Union, Optional, Dict, Any


class CLMaskPretrainer(Pretrainer):
    """
    Pretrain a model on an input with character-level masking

    This task is to take an input, apply noisy transformations
    (including the mask), and predict the original input

    Every character in the input has a `character_mask_rate` probability of
    being masked (independent of each other). If `consecutive_mask_len` is
    greater than zero, one consecutive sequence of length `consecutive_mask_len`
    will also be masked.
    """

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        preprocessor: Optional[Preprocessor] = None,
        task_name="pretraining",
        load_from: str = None,
        embedder_args: Dict[str, Any] = {},
        augmentor: Optional[Augmentor] = None,
        character_mask_rate: float = 0.125,
        consecutive_mask_len: int = 5,
        n_transformer_layers: int = 3,
        n_transformer_heads: int = 8,
        ff_dim: int = 128,
        **kwargs,
    ):
        self.save_attr = [
            "task_name",
            "character_mask_rate",
            "consecutive_mask_len",
            "n_transformer_layers",
            "n_transformer_heads",
            "ff_dim",
        ]
        self.augmentor = augmentor
        self.character_mask_rate = character_mask_rate
        self.consecutive_mask_len = consecutive_mask_len
        self.n_transformer_layers = n_transformer_layers
        self.n_transformer_heads = n_transformer_heads
        self.ff_dim = ff_dim

        super().__init__(
            embedder=embedder,
            preprocessor=preprocessor,
            task_name=task_name,
            load_from=load_from,
            embedder_args=embedder_args,
            **kwargs,
        )

    def get_model(self):
        """
        Transformer layers on top of the base
        """
        emb_out_shape = self.embedder.model.outputs[0].shape[-1]
        transformer_layers = [
            TransformerBlock(
                emb_out_shape,
                self.n_transformer_heads,
                self.ff_dim,
            )
            for _ in range(self.n_transformer_layers)
        ]
        n_characters = self.preprocessor.n_chars + 1
        inp = self.embedder.model.input
        x = self.embedder.model.output

        x = PositionEmbedding(self.preprocessor.max_example_len, emb_out_shape)(x)
        for layer in transformer_layers:
            x = layer(x)
        dense = tf.keras.layers.Dense(
            n_characters + 1, activation="softmax", dtype="float32"
        )
        return tf.keras.Model(inp, dense(x))

    def get_mask(self, seq_len):
        mask = np.zeros((seq_len,))
        max_ind = seq_len - self.consecutive_mask_len
        if max_ind <= 0:
            return mask
        sampled_ind = np.random.randint(0, max_ind)
        mask[sampled_ind : sampled_ind + self.consecutive_mask_len] = 1
        return mask

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
                    np.random.rand(*x_str.shape) < self.character_mask_rate,
                    self.preprocessor.mask_token_ind,
                    x_str,
                )
                if self.consecutive_mask_len > 0:
                    x_str = np.where(
                        self.get_mask(len(x_str)) == 1,
                        self.preprocessor.mask_token_ind,
                        x_str,
                    )
                x.append(x_str)
                y.append(y_str)
                if len(x) == batch_size or n + 1 == len(data):
                    yield np.array(x), np.array(y)
                    x, y = [], []

    def evaluate_prediction(self, x: np.ndarray, pred: np.ndarray, prep: Preprocessor):
        """
        evaluate the predicted string compared to the input
        """
        out_str = ""
        rev = prep.char_rev
        maxes = np.argmax(pred, axis=2)
        for n in range(len(x)):
            out_str += f"Example {n}:\n"
            out_str += "".join(rev[i] for i in x[n]) + "\n"
            out_str += "".join(rev[i] for i in maxes[n]) + "\n\n"
        return out_str

    def save(self, path: str):
        """
        Persist the model weights and the configuration necessary to load it back up
        """
        base_path = os.path.join(path, self.task_name)
        os.makedirs(base_path, exist_ok=True)
        out_dict = {}
        for attr in self.save_attr:
            out_dict[attr] = getattr(self, attr)
        with open(os.path.join(base_path, "config.json"), "w") as f:
            json.dump(out_dict, f, indent=2)
        weights_path = os.path.join(base_path, "weights")
        self.model.save_weights(weights_path)

    def load(self, path: str):
        """
        Load the persisted model weights and attributes
        """
        with open(os.path.join(path, self.task_name, "config.json"), "r") as f:
            config = json.load(f)
        for attr, value in config.items():
            setattr(self, attr, value)

        weights_path = os.path.join(path, self.task_name, "weights")
        self.model = self.get_model()
        self.model.load_weights(weights_path)


class CLMaskPretrainerEvaluationCallback(tf.keras.callbacks.Callback):
    def __init__(
        self, data: List[str], pretrainer: CLMaskPretrainer, log_every: int = 5
    ):
        super().__init__()
        self.pretrainer = pretrainer
        self.log_every = log_every
        self.gen = pretrainer.generator(data)

    def on_epoch_end(self, epoch, logs):
        if epoch % self.log_every == 0:
            x, y = next(self.gen)
            log_str = self.pretrainer.evaluate_prediction(
                x, self.model(x), self.pretrainer.preprocessor
            )
            print(log_str)
        return
