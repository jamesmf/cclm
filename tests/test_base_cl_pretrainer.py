import pytest
from cclm.pretrainers.base_cl_pretrainer import (
    BasePretrainer,
    BasePretrainerEvaluationCallback,
)
from cclm.preprocessing import Preprocessor
from cclm.models import CCLMModelBase
import numpy as np
import tensorflow as tf


def set_seed():
    np.random.seed(0)


CORPUS = [
    "hello i am a test string",
    "hi there I am also a test string",
    "this is more words and the length is longer",
    "here is another for us to test the",
    "vocabulary and in order for ther eto be enough sampled values for the tensorflow log uniform candidate sampler",
]


def test_bp_init():
    prep = Preprocessor(max_example_len=10)
    prep.fit(CORPUS)
    base = CCLMModelBase(preprocessor=prep)
    pretrainer = BasePretrainer(base)

    assert True, "error initializing a BasePretrainer"


def test_bp_fit():
    prep = Preprocessor(max_example_len=10)
    prep.fit(CORPUS)
    base = CCLMModelBase(preprocessor=prep)
    pretrainer = BasePretrainer(base)
    gen = pretrainer.generator(CORPUS)
    pretrainer.model.compile("adam", tf.keras.losses.SparseCategoricalCrossentropy())
    pretrainer.model.fit(gen, steps_per_epoch=2, epochs=2)

    assert True, "error fitting BasePretrainer"