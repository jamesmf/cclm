import pytest
from cclm.pretrainers import MaskedLanguagePretrainer
from cclm.preprocessing import Preprocessor
from cclm.models import Embedder
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


def test_fit():
    prep = Preprocessor(max_example_len=10)
    prep.fit(CORPUS)
    emb = Embedder(prep.max_example_len, prep.n_chars)
    mlp = MaskedLanguagePretrainer(
        embedder=emb,
        preprocessor=prep,
        stride_len=1,
        downsample_factor=1,
        n_strided_convs=2,
        training_pool_mode="global",
    )
    gen = mlp.generator(CORPUS, batch_size=2)
    mlp.pretraining_model.compile("adam", "categorical_crossentropy")
    x = next(gen)[0]
    print(x[0])
    print(x[1])
    print(x[2])
    print(x[2].shape)
    print(mlp.pretraining_model.summary())
    result = mlp.pretraining_model.predict(x)
    print(result)
    mlp.pretraining_model.fit(gen, epochs=1, steps_per_epoch=2)
    assert True, "error in MLMPreprocessor.fit()"


def test_get_substr_short():
    test_str = "hello"

    prep = Preprocessor(max_example_len=10)
    prep.fit(CORPUS)
    emb = Embedder(prep.max_example_len, prep.n_chars)
    mlp = MaskedLanguagePretrainer(
        embedder=emb,
        preprocessor=prep,
    )
    assert (
        mlp.get_substr(test_str) == test_str
    ), "string with len() < self.preprocessor.max_example_len should be substring'd to the same string"


def test_get_substr_long():
    test_str = "hello i am a string longer than 10 characters"
    set_seed()
    prep = Preprocessor(max_example_len=10)
    prep.fit(CORPUS)
    emb = Embedder(prep.max_example_len, prep.n_chars)
    mlp = MaskedLanguagePretrainer(embedder=emb, preprocessor=prep)
    assert mlp.get_substr(test_str) == "hello i am"


def test_batch_from_strs():
    set_seed()

    prep = Preprocessor(max_example_len=16)
    prep.fit(CORPUS)
    emb = Embedder(prep.max_example_len, prep.n_chars)
    mlp = MaskedLanguagePretrainer(
        embedder=emb,
        preprocessor=prep,
    )
    inps, spans, outs = mlp.batch_from_strs(CORPUS)
    print(inps)
    print(spans)
    print(outs)
    assert inps[0] == "o i am a ???? st", "unexpected MLM example in batch"
    assert spans[0][0] == 9, "unexpected token start index"
    assert spans[0][1] == 13, "unexpected token end index"
    assert spans[0][2] == "test", "unexpected masked value"
    assert outs[0] == 37, "unexpected return index"
