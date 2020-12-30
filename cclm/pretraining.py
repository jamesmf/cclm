import numpy as np
import json
import os

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


class Pretrainer:
    """
    A pretrainer needs to accept a base, implement some core layers that it
    will fit (in addition to the base, optionally), and implement a top to the
    network that represents its specific task.

    The
    """

    def __init__(self, base=None, task_name="pretraining", base_args={}, **kwargs):
        self.transfer_layer_names = []
        self.model = self.get_model()
        if base is None:
            base = CCLMModelBase(**base_args)
        self.base = base
        self.task_name = task_name

    def fit(self, *args, **kwargs):
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


class MaskedLanguagePretrainer(Pretrainer):
    def __init__(self, *args, **kwargs):
        self.preprocessor = kwargs.get("preprocessor")
        base = kwargs.get("base")
        if base:
            self.preprocessor = base.preprocessor
        super().__init__(*args, **kwargs)

    def get_model(self, n_conv_filters: int = 128):
        """
        Until handled better, inputs need to be padded to a multiple of filter_stride_len
        """
        ln = f"mlm_core_c1"
        self.transfer_layer_names.append(ln)
        # reduce the size, transformer, upsample
        filter_stride_len = 4
        model = tf.keras.Sequential(
            [
                Conv1D(
                    n_conv_filters,
                    filter_stride_len,
                    padding="same",
                    activation="tanh",
                ),
                Conv1D(
                    n_conv_filters,
                    filter_stride_len,
                    strides=filter_stride_len,
                    padding="same",
                    activation="tanh",
                ),
                TransformerBlock(embed_dim=n_conv_filters),
                tf.keras.layers.UpSampling1D(size=filter_stride_len),
            ]
        )

        return model
