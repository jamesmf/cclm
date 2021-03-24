import pytest
from cclm.pretraining import MaskedLanguagePretrainer
from cclm.preprocessing import MLMPreprocessor
from cclm.models import CCLMModelBase
import numpy as np


def set_seed():
    np.random.seed(0)


CORPUS = [
    "hello i am a test string",
    "hi there I am also a test string",
    "this is more words and the length is longer",
    "here is another for us to test the",
    "vocabulary and in order for ther eto be enough sampled values for the tensorflow log uniform candidate sampler",
]


def test_fit():
    prep = MLMPreprocessor(max_example_len=10)
    prep.fit(CORPUS)
    base = CCLMModelBase(preprocessor=prep)
    mlp = MaskedLanguagePretrainer(base=base)
    mlp.fit(CORPUS, epochs=1, batch_size=2)
    assert False, "error in MLMPreprocessor.fit()"


def test_freeze():
    prep = MLMPreprocessor(max_example_len=10)
    prep.fit(CORPUS)
    base = CCLMModelBase(preprocessor=prep)
    mlp = MaskedLanguagePretrainer(base=base)
    mlp.fit(CORPUS, epochs=1, batch_size=2)

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
    mlp.fit(CORPUS, epochs=1, batch_size=2)  # fit easy way to build model weights

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


def test_get_substr_short():
    test_str = "hello"

    prep = MLMPreprocessor(max_example_len=10)
    prep.fit(CORPUS)
    base = CCLMModelBase(preprocessor=prep)
    mlp = MaskedLanguagePretrainer(base=base)
    assert (
        mlp.get_substr(test_str) == test_str
    ), "string with len() < self.preprocessor.max_example_len should be substring'd to the same string"


def test_get_substr_long():
    test_str = "hello i am a string longer than 10 characters"
    set_seed()
    prep = MLMPreprocessor(max_example_len=10)
    prep.fit(CORPUS)
    base = CCLMModelBase(preprocessor=prep)
    mlp = MaskedLanguagePretrainer(base=base)
    assert mlp.get_substr(test_str) == "hello i am"


def test_batch_from_strs():
    set_seed()

    prep = MLMPreprocessor(max_example_len=16)
    prep.fit(CORPUS)
    base = CCLMModelBase(preprocessor=prep)
    mlp = MaskedLanguagePretrainer(base=base)
    inps, spans, outs = mlp.batch_from_strs(CORPUS)
    print(inps)
    print(spans)
    print(outs)
    assert inps[0] == "o i am a ???? st", "unexpected MLM example in batch"
    assert spans[0][0] == 9, "unexpected token start index"
    assert spans[0][1] == 13, "unexpected token end index"
    assert spans[0][2] == "test", "unexpected masked value"
    assert outs[0] == 36, "unexpected return index"


# def test_batch_output_index_matches_inp_str():
#     set_seed()

#     prep = MLMPreprocessor(max_example_len=16)
#     prep.fit(CORPUS)
#     base = CCLMModelBase(preprocessor=prep)
#     mlp = MaskedLanguagePretrainer(base=base)
#     inps, spans, outs = mlp.batch_from_strs(CORPUS)
#     output = outs[0]
