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

MAX_LEN = 10
PREP_ARGS = {"max_example_len": MAX_LEN}
CLM_ARGS = {"n_transformer_layers": 3, "n_transformer_heads": 2, "ff_dim": 4}
EMB_ARGS = {"max_len": MAX_LEN}


@pytest.fixture(scope="module")
def cml():
    prep = Preprocessor(**PREP_ARGS)
    prep.fit(CORPUS)
    emb = Embedder(**EMB_ARGS, n_chars=prep.n_chars)
    pretrainer = CLMaskPretrainer(emb, preprocessor=prep, **CLM_ARGS)
    return pretrainer


def test_clmaskpretrainer_fit(cml):
    prep = cml.preprocessor
    pretrainer = cml
    gen = pretrainer.generator(CORPUS)
    x, y = next(gen)
    print(pretrainer.model.predict(x))
    pretrainer.model.compile("adam", tf.keras.losses.SparseCategoricalCrossentropy())
    print(pretrainer.model.summary())
    print(prep.n_chars)
    print(next(gen))
    pretrainer.model.fit(gen, steps_per_epoch=2, epochs=2)

    assert True, "error fitting CLMaskPretrainer"


def test_clmask_loading(tmp_path, cml):
    sub = tmp_path / "clmask_load_test"
    sub.mkdir()
    prep = cml.preprocessor
    cmp = cml
    cmp.save(sub)
    emb_args = dict(max_len=prep.max_example_len, n_chars=prep.n_chars)
    cmp2 = CLMaskPretrainer(preprocessor=prep, embedder_args=emb_args, **CLM_ARGS)
    cmp2.load(sub)
    cmp3 = CLMaskPretrainer(
        preprocessor=prep, load_from=sub, embedder_args=emb_args, **CLM_ARGS
    )
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


def test_clmask_pretrainer_filter(cml):
    cml.filters = [lambda x: "vocab" in x]
    gen = cml.generator(CORPUS)
    x, y = next(gen)
    print(x.shape)
    assert x.shape[0] == len(CORPUS) - 1, "filter not applied in CLMaskPretrainer.fit()"
