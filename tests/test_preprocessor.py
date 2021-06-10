import pytest
from cclm.preprocessing import Preprocessor

CORPUS = [
    "hello i am a test string",
    "hi there I am also a test string",
    "this is more words and the length is longer",
    "here is another for us to test the",
    "vocabulary and in order for ther eto be enough sampled values for the tensorflow log uniform candidate sampler",
]


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