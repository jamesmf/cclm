from typing import List
import tensorflow as tf
import numpy as np
from ..preprocessing import Preprocessor
from ..models import CCLMModelBase, TransformerBlock


class Pretrainer:
    """
    A pretrainer needs to accept a base, implement some core layers that it
    will fit (in addition to the base, optionally), and implement a top to the
    network that represents its specific task.
    """

    def __init__(
        self,
        base=None,
        task_name="pretraining",
        load_from: str = None,
        base_args={},
        **kwargs,
    ):
        self.preprocessor = kwargs.get("preprocessor", Preprocessor())

        if base is None:
            base = CCLMModelBase(**base_args)
        self.base = base
        if load_from:
            self.model = tf.keras.models.load_model(load_from)
        else:
            self.model = self.get_model()
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
        self.model.trainable = False

    def unfreeze(self):
        """
        Make the layers in the model not trainable.
        """
        self.model.trainable = True

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
