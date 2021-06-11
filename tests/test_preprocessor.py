import pytest
import numpy as np
from cclm.preprocessing import Preprocessor

CORPUS = [
    "hello i am a test string",
    "hi there I am also a test string",
    "this is more words and the length is longer",
    "here is another for us to test the",
    "vocabulary and in order for ther eto be enough sampled values for the tensorflow log uniform candidate sampler",
]


def reverse(prep: Preprocessor, x: np.ndarray):
    return "".join(prep.char_rev[i] for i in x)


def test_prep_encode_string():
    prep = Preprocessor(max_example_len=10)
    prep.fit(CORPUS)
    my_string = CORPUS[0]
    example = prep.string_to_array(my_string, 5)
    assert (
        0 not in example
    ), "0 present when encoding string whose characters should all be in char_dict"
    assert len(example) == 5, "shape of string_to_array incorrect"


def test_prep_fit_char_dict():
    prep = Preprocessor()
    prep.fit(["a a", "b a"], min_char_count=2)
    print(prep.char_dict)
    assert "a" in prep.char_dict, "char dict not fit properly"
    assert "b" not in prep.char_dict, "char dict contains characters below min value"


def test_save_load(tmp_path):
    mel = 21
    p = Preprocessor(max_example_len=mel)
    p.save(tmp_path)
    saved = tmp_path / "cclm_config.json"
    p2 = Preprocessor(load_from=saved)
    assert p2.max_example_len == mel


def test_default_tokenizer_behavior(tmp_path):
    p = Preprocessor()
    p.fit(CORPUS)
    assert (
        "string" in p.tokenizer.get_vocab()
    ), "fit tokenizer does not have expected tokens"


def test_preprocessor_downsample_when_too_short(tmp_path):
    p = Preprocessor(downsample_factor=4)
    p.fit(CORPUS)
    # a string that is not long enough
    s = "hi"
    t = p.string_to_array(s)
    assert (
        t.shape[0] == 4
    ), "Preprocessor created a sequence too short when processing a string less then downsample_factor"


def test_preprocessor_downsample_default(tmp_path):
    p = Preprocessor(downsample_factor=4)
    p.fit(CORPUS)
    # a string that is not long enough
    s = "i am 6"
    t = p.string_to_array(s)
    assert (
        t.shape[0] == 8
    ), "Preprocessor(downsample_factor=4) should pad a string of len(6) to sequence length 8"


def test_preprocessor_cls():
    p = Preprocessor(add_cls=True)
    p.fit(CORPUS)
    # a string that is not long enough
    s = "a short string"
    arr = p.string_to_array(s)
    reversed = reverse(p, arr)
    assert (
        reversed == "a short string[CLS]"
    ), "expected [CLS] token to be added when add_cls=True on Preprocessor"


def test_preprocessor_cls_and_downsample():
    p = Preprocessor(add_cls=True, downsample_factor=16)
    p.fit(CORPUS)
    # a string that is not long enough, we should expect [CLS] in arr[1]
    s = "a"
    arr = p.string_to_array(s)
    assert (
        arr[1] == p.cls_token_ind
    ), "expected [CLS] token ind to be added directly after the sequence when downsample_factor > len(string)"
    assert arr[2] == 0, "zero-padding not as expected when add_cls == True"


def test_preprocessor_cls_long_string_no_specific_length():
    p = Preprocessor(add_cls=True)
    p.fit(CORPUS)
    # a string that exactly the max_example_len
    s = "char"
    arr = p.string_to_array(s)
    reversed = reverse(p, arr)
    assert (
        arr[-1] == p.cls_token_ind
    ), "expected [CLS] token ind to be added in the last index"
    assert (
        reversed == "char[CLS]"
    ), "expected [CLS] token ind to be added in the last index"


def test_preprocessor_cls_long_string_specific_length():
    p = Preprocessor(add_cls=True)
    p.fit(CORPUS)
    # a string that exactly the max_example_len
    s = "char"
    arr = p.string_to_array(s, length=4)
    reversed = reverse(p, arr)
    assert (
        arr[-1] == p.cls_token_ind
    ), "expected [CLS] token ind to be added in the last index"
    assert (
        reversed == "cha[CLS]"
    ), "expected [CLS] token ind to be added in the last index"


def test_preprocessor_cls_long_string_downsample():
    p = Preprocessor(add_cls=True, max_example_len=5, downsample_factor=4)
    p.fit(CORPUS)
    # a string that exactly the max_example_len
    s = "char"
    arr = p.string_to_array(s)
    reversed = reverse(p, arr)
    assert (
        reversed == "cha[CLS]"
    ), "expected [CLS] token ind to be added in the last index"
