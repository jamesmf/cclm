from os import PRIO_PGRP
import pytest
from cclm.pretrainers.cl_mask_pretrainer import (
    CLMaskPretrainer,
)
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

PREP_ARGS = {"max_example_len": 10}


def test_clmaskpretrainer_init():
    prep = Preprocessor(max_example_len=10)
    prep.fit(CORPUS)
    emb = Embedder(max_len=prep.max_example_len, n_chars=prep.n_chars)
    pretrainer = CLMaskPretrainer(emb, preprocessor=prep)

    assert True, "error initializing a CLMaskPretrainer"


def test_clmaskpretrainer_fit():
    prep = Preprocessor(max_example_len=10)
    prep.fit(CORPUS)
    emb = Embedder(prep.max_example_len, prep.n_chars, n_blocks=2)
    pretrainer = CLMaskPretrainer(emb, preprocessor=prep, n_transformer_layers=1)
    gen = pretrainer.generator(CORPUS)
    x, y = next(gen)
    print(emb.model.predict(x))
    print(pretrainer.model.predict(x))
    pretrainer.model.compile("adam", tf.keras.losses.SparseCategoricalCrossentropy())
    print(pretrainer.model.summary())
    print(prep.n_chars)
    print(next(gen))
    pretrainer.model.fit(gen, steps_per_epoch=2, epochs=2)

    assert True, "error fitting CLMaskPretrainer"


def test_clmask_loading(tmp_path):
    sub = tmp_path / "clmask_load_test"
    sub.mkdir()
    prep = Preprocessor(**PREP_ARGS)
    prep.fit(CORPUS)
    cmp = CLMaskPretrainer(preprocessor=prep)
    cmp.save(sub)
    cmp2 = CLMaskPretrainer(preprocessor=prep)
    cmp2.load(sub)
    cmp3 = CLMaskPretrainer(preprocessor=prep, load_from=sub)
    w1 = cmp.model.get_weights()[0]
    w2 = cmp2.model.get_weights()[0]
    w3 = cmp3.model.get_weights()[0]
    assert np.allclose(
        w1, w2
    ), "failed to load weights of CLMaskPretrainer model using .load()"
    assert np.allclose(
        w1, w3
    ), "failed to load weights of CLMaskPretrainer model using load_from="
    for attr in cmp.save_attr:
        att1 = getattr(cmp, attr)
        att2 = getattr(cmp2, attr)
        att3 = getattr(cmp2, attr)
        assert att1 == att2, f"failed to set attribute {attr} when using .load()"
        assert att1 == att3, f"failed to set attribute {attr} when using load_from="