import pytest
from cclm import Preprocessor

CORPUS = ["hello i am a test string", "hi there I am also a test string"]


def test_prep_string_to_example():
    prep = Preprocessor(max_example_len=10, max_token_len=10, negative_samples=2)
    prep.fit(CORPUS)
    my_string = CORPUS[0]
    example = prep.string_to_example(my_string, CORPUS)
    print(example)
    assert example[0].shape == (3, 10), "Left side encoded to right shape"
    assert example[1].shape == (
        3,
    ), f"Right side not encoded to proper shape, got {example[1].shape}"


def test_prep_fit_vocab_table():
    prep = Preprocessor(vocab_table_size=4, max_word_mask=1)
    prep.fit(["a a", "b a"])
    print(prep.vocab_table)
    assert prep.vocab_table.count("a") == 3, "Vocab table not proportioned properly"
    assert prep.vocab_table.count("b") == 1, "Vocab table not proportioned properly"


def test_prep_fit_vocab_table_ngram():
    prep = Preprocessor(vocab_table_size=6, max_word_mask=2)
    prep.fit(["a a", "b a"])
    print(prep.vocab_table)
    assert prep.vocab_table.count("a") == 3, "Vocab table not proportioned properly"
    assert prep.vocab_table.count("b") == 1, "Vocab table not proportioned properly"
    assert prep.vocab_table.count("a a") == 1, "Vocab table not proportioned properly"
    assert prep.vocab_table.count("b a") == 1, "Vocab table not proportioned properly"


def test_prep_tokenizer_strip():
    prep = Preprocessor()
    result = prep.tokenize("  the string has a newline\n")
    print(result)
    assert len(result) == 5, "tokenizer returning incorrect number of tokens"
    assert "" not in result, "blanks in tokenized result"
    assert " " not in result, "spaces in tokenized result"


def test_prep_scrub_punctuation():
    prep = Preprocessor()
    result = prep.scrub("hello,")
    assert result == "hello", ".scrub() didn't remove ','"
    result = prep.scrub("Hello")
    assert result == "hello", ".scrub() didn't lowercase"
    result = prep.scrub("hello!")
    assert result == "hello", ".scrub() didn't remove '!'"
    result = prep.scrub("hello?")
    assert result == "hello", ".scrub() didn't remove '?'"
    result = prep.scrub("@hello")
    assert result == "hello", ".scrub() didn't remove '@' at beginning"


def test_prep_string_to_example_return_val():
    prep = Preprocessor(batch_size=1)
    prep.fit(CORPUS)
    result = prep.string_to_example(CORPUS[0], CORPUS, return_example=True)
    print(result[0])
    print(result[1])
    print(result[2])
    print(result)
    assert len(result) == 3, "return_example not working in string_to_example"
    assert len(result[2]) == 3


def test_prep_examples_generator():
    bsize = 2
    prep = Preprocessor(batch_size=bsize, negative_samples=1, max_example_len=16)
    prep.fit(CORPUS)
    gen = prep.examples_generator(CORPUS)
    result = next(gen)
    print(result)
    assert len(result[0]) == 2 * bsize, "examples_generator X is not proper shape"
    assert (
        result[0][0].shape[0] == prep.max_example_len
    ), "examples_generator not returning proper example length"


def test_save_load(tmp_path):
    mel = 21
    p = Preprocessor(max_example_len=mel)
    p.save(tmp_path)
    saved = tmp_path / "ecle_config.json"
    p2 = Preprocessor(load_from=saved)
    assert p2.max_example_len == mel
