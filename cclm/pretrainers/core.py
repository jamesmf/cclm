from typing import List, Optional
import tensorflow as tf
import numpy as np
from ..preprocessing import Preprocessor
from ..models import Embedder


class Pretrainer:
    """
    A pretrainer needs to accept a base, implement some core layers that it
    will fit (in addition to the base, optionally), and implement a top to the
    network that represents its specific task.
    """

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        preprocessor: Optional[Preprocessor] = None,
        task_name: str = "pretraining",
        load_from: str = None,
        embedder_args={},
        **kwargs,
    ):
        if preprocessor is None:
            preprocessor = Preprocessor()
        self.preprocessor = preprocessor
        self.task_name = task_name

        if embedder is None:
            embedder = Embedder(**embedder_args)
        self.embedder = embedder
        if load_from:
            self.load(load_from)
        else:
            self.model = self.get_model()

    def get_model(self, *args, **kwargs):
        """
        This should return a Model that accepts input with the shape of a the embedder and
        produces output of the same shape.
        """

    def load(self, path: str):
        pass

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
