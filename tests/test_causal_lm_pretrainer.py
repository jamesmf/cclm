import pytest
import cclm.pretrainers.causal_lm as clm
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
