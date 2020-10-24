import numpy as np
import json
import os

# from tf.keras.callbacks import ModelCheckpoint
# from keras.layers import (
#     Input,
#     SeparableConv1D,
#     Conv1D,
#     MaxPooling1D,
#     Lambda,
#     Multiply,
#     Dropout,
#     Dense,
#     Embedding,
#     Dot,
#     Concatenate,
#     Reshape,
#     Add,
#     Average,
#     GlobalMaxPooling1D,
#     GlobalMaxPooling2D,
#     Permute,
#     RepeatVector,
#     Flatten,
# )
# from keras.layers.wrappers import TimeDistributed
# from keras.regularizers import l1_l2
# from keras.optimizers import Adam, RMSprop, SGD
# from keras.models import Model, load_model
# from keras.initializers import TruncatedNormal
# from keras.constraints import MinMaxNorm
# import keras.backend as K
import tensorflow as tf
from .preprocessing import Preprocessor
from .models import CCLMModelBase


class Pretrainer:
    """
    A pretrainer needs to accept a base, implement some core layers that it
    will fit (in addition to the base, optionally), and implement a top to the
    network that represents its specific task.

    The
    """

    def __init__(self, base=None, task_name="pretraining", base_args={}, **kwargs):
        self.transfer_layer_names = []
        self.model = None
        self.extra_inputs = []
        self.extra_outputs = []
        if base is None:
            base = CCLMModelBase(**base_args)
        self.base = base
        self.task_name = task_name
        self.common_output = self.add_core_layers()
        self.specific_output = self.add_task_specific_layers()
        outputs = (
            self.specific_output
            if isinstance(self.specific_output, list)
            else [self.specific_output]
        )
        print(self.base.embedder.input, self.extra_inputs)
        print(outputs)
        self.model = Model(
            self.base.embedder.inputs + self.extra_inputs, outputs + self.extra_outputs
        )

    def fit(self):
        raise NotImplementedError

    def compile(self, *args, **kwargs):
        self.model.compile(*args, **kwargs)


class MaskedLanguagePretrainer(Pretrainer):
    def __init__(self, *args, **kwargs):
        # preprocessor = kwargs.get("preprocessor", None)
        # if preprocessor is None:
        #     assert (
        #         "preprocessor_args" in kwargs
        #     ), "Must pass proprocessor_args if not passing preprocessor"
        #     preprocessor = Preprocessor(**kwargs["preprocessor_args"])
        self.preprocessor = kwargs["preprocessor"]
        super().__init__(*args, **kwargs)

    def add_core_layers(self):
        """
        Add a number of conv layers to go from a character level representation
        to a deeper representation, then add attention heads over it. Use
        dilation_rate to increase the ability of the model to increase in scope
        """
        tn = self.task_name
        ln = f"{tn}_core_c1"
        self.transfer_layer_names.append(ln)
        new_conv = Conv1D(
            128, 3, padding="same", activation="tanh", name=ln, dilation_rate=2
        )(self.base.embedder.output)
        new_conv = Dropout(0.25)(new_conv)
        # add attention heads
        n_attention_heads = 4
        new_conv_shape = int(new_conv.shape[1])
        att_heads = []

        for head_num in range(n_attention_heads):
            att = Permute((2, 1))(new_conv)
            ln = f"{tn}_core_att_{head_num}"
            self.transfer_layer_names.append(ln)
            att = Dense(new_conv_shape, activation="softmax", name=ln)(att)
            att = Permute((2, 1))(att)
            att_out = Multiply()([new_conv, att])
            att_heads.append(att_out)
        cat = Concatenate()(att_heads)

        ln = f"{tn}_core_c2"
        self.transfer_layer_names.append(ln)
        conv_2 = Conv1D(
            128, 3, padding="same", activation="relu", name=ln, dilation_rate=3
        )(cat)
        ln = f"{tn}_core_c3"
        self.transfer_layer_names.append(ln)
        conv_3 = Conv1D(
            128, 3, padding="same", activation="relu", name=ln, dilation_rate=3
        )(conv_2)
        return conv_3

    def add_task_specific_layers(self):
        """
        Take the representation we've learned and put a task-specific head on
        it to do masked language modeling.
        """
        gmp = GlobalMaxPooling1D()(self.common_output)
        d = Dense(
            self.preprocessor.vocab_size,
            activation="softmax",
            name=f"{self.task_name}_final_dense",
        )(gmp)
        return d
