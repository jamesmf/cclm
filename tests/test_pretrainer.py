import pytest
from cclm.pretraining import MaskedLanguagePretrainer
from cclm.preprocessing import MLMPreprocessor
from cclm.models import CCLMModelBase
import numpy as np


CORPUS = [
    "hello i am a test string",
    "hi there I am also a test string",
    "this is more words and the length is longer",
    "here is another for us to test the",
    "vocabulary and in order for ther eto be enough sampled values for the tensorflow log uniform candidate sampler",
]


def test_freeze():
    prep = MLMPreprocessor(max_example_len=10)
    prep.fit(CORPUS)
    base = CCLMModelBase(preprocessor=prep)
    mlp = MaskedLanguagePretrainer(base=base)
    mlp.fit(CORPUS, epochs=1)

    mlp.freeze()
    mean = np.mean(
        [np.mean(i[0]) for i in mlp.model.get_weights() if isinstance(i[0], np.ndarray)]
    )
    print(mean)
    mlp.fit(CORPUS, epochs=1)
    mean_new = np.mean(
        [np.mean(i[0]) for i in mlp.model.get_weights() if isinstance(i[0], np.ndarray)]
    )
    assert mean == mean_new, "freeze did not work, weights changed"


def test_unfreeze():
    prep = MLMPreprocessor(max_example_len=10)
    prep.fit(CORPUS)
    base = CCLMModelBase(preprocessor=prep)
    mlp = MaskedLanguagePretrainer(base=base)
    mlp.fit(CORPUS, epochs=1)  # fit easy way to build model weights

    mean = np.mean(
        [np.mean(i[0]) for i in mlp.model.get_weights() if isinstance(i[0], np.ndarray)]
    )
    print(mean)
    mlp.fit(CORPUS, epochs=1)
    mlp.fit(CORPUS, epochs=5, print_interval=1)
    mean_new = np.mean(
        [np.mean(i[0]) for i in mlp.model.get_weights() if isinstance(i[0], np.ndarray)]
    )
    assert mean != mean_new, "unfreeze did not work, weights remained the same"