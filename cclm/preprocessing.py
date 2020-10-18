from tqdm import tqdm
import numpy as np
import json
import os
import re
from collections import Counter

PUNCT = '~!@#$%^&*()-+=_.,?":;' + "'"


class Preprocessor:
    def __init__(
        self,
        load_from=None,
        vocab_size=10000,
        max_example_len=128,
        batch_size=16,
        num_stopwords=250,
        mask_output_len=4,
    ):
        self.char_dict = {}
        self.char_rev = {}
        self.token_dict = {}
        self.token_rev = {}
        self.vocab_size = vocab_size
        self.max_example_len = max_example_len
        self.batch_size = batch_size
        self.num_stopwords = num_stopwords
        self.mask_output_len = mask_output_len
        if load_from:
            self._load(load_from)

    def fit(self, data, min_char_freq=1, progbar=False):
        """
        Create a character-level dictionary based on a list of strings
        """
        char_counter = Counter()
        token_counter = Counter()
        if progbar:
            iterator_ = tqdm(data)
        else:
            iterator_ = data
        for example in iterator_:
            chars = Counter(example)
            split = self.tokenize(example)
            tokens = Counter(split)
            # get counts of characters and tokens
            for char, char_count in chars.items():
                try:
                    char_counter[char] += char_count
                except KeyError:
                    char_counter[char] = char_count
            for token, token_count in tokens.items():
                token = self.scrub(token)
                try:
                    token_counter[token] += token_count
                except KeyError:
                    token_counter[token] = token_count
        token_dict = dict(token_counter)
        token_counts = sorted(token_dict.items(), key=lambda x: x[1], reverse=True)
        token_counts = token_counts[
            self.num_stopwords : self.vocab_size + self.num_stopwords
        ]

        counts = [k for k, v in char_counter.items() if v >= min_char_freq]
        self.char_rev = {0: "", 1: "?", 2: "?", 3: ""}
        self.token_rev = {}
        for c in sorted(counts):
            n = len(self.char_rev)
            self.char_rev[n] = c
            self.char_dict[c] = n
        for w, w_count in sorted(token_counts):
            n = len(self.token_rev)
            self.token_rev[n] = w
            self.token_dict[w] = n

    def scrub(self, token):
        """
        Normalize a token by removing punctuation. Used to build a vocabulary
        and to choose tokens to mask during pretraining.
        """
        token = token.lower()
        while len(token) > 0 and token[0] in PUNCT:
            token = token[1:]
        while len(token) > 0 and token[-1] in PUNCT:
            token = token[:-1]
        token = re.sub("\d", "#", token)
        return token

    def tokenize(self, string_to_tokenize):
        string_to_tokenize = string_to_tokenize.lower().strip()
        return [
            tok.strip() for tok in string_to_tokenize.split(" ") if tok.strip() != ""
        ]

    def string_to_array(self, string_in, length, padding_pre=False):
        # truncate
        if padding_pre:
            s = string_in[-length:]
        else:
            s = string_in[:length]
        # map char -> int and left-zero-pad
        mapped = list(map(lambda x: self.char_dict.get(x, 1), s))
        if padding_pre:
            r = np.pad(mapped, (length - len(s), 0), "constant", constant_values=(0, 0))
        else:
            r = np.pad(mapped, (0, length - len(s)), "constant", constant_values=(0, 0))
        return r

    def string_to_example(
        self, example, return_example=False, allow_null_examples=False
    ):
        # simple tokenization
        sp = [tok.strip() for tok in example.split(" ") if tok.strip() != ""]
        # normalize to see what we can replace
        normed = [self.scrub(tok) for tok in sp]
        # see which tokens are in the vocabulary
        replaceable_tokens = [
            (t, i) for i, t in enumerate(normed) if t in self.token_dict
        ]
        assert (
            len(sp) >= 2
        ), "minimum length of an example is 2 tokens (white-space delimited)"
        assert (
            len(replaceable_tokens) > 0 or allow_null_examples
        ), f"called string_to_example on string with no tokens that are in the vocabulary and allow_null_examples=True\n{example}"
        if len(replaceable_tokens) == 0 and allow_null_examples:
            return None
        # choose a token to replace
        rep_ind = np.random.randint(0, len(replaceable_tokens))

        rep, rep_ind = replaceable_tokens[rep_ind]

        # get the index of the token for the output
        mask_ind = self.token_dict[rep]
        label_array = np.zeros(self.vocab_size)
        label_array[mask_ind] = 1

        # piece the masked input back together
        left = " ".join(sp[:rep_ind])
        right = " ".join(sp[rep_ind + 1 :]) if len(sp) > rep_ind + 1 else ""
        left_len = len(left)
        right_len = len(right)
        thresh = (self.max_example_len - self.mask_output_len - 2) // 2
        right_diff = thresh - left_len if left_len < thresh else 0
        left_diff = thresh - right_len if right_len < thresh else 0
        left_sub = left[-(thresh + left_diff) :]
        right_sub = right[: (thresh + right_diff)]
        combo = left_sub + " " + "?" * 4 + " " + right_sub
        encoded = self.string_to_array(combo, self.max_example_len)
        ret_val = [encoded, label_array]
        if return_example:
            ret_val += [combo]
        return ret_val

    def strings_to_examples(self, strings):
        enc = np.zeros((len(strings), self.max_example_len))
        labels = np.zeros((len(strings), self.vocab_size))
        for n in range(len(strings)):
            enc[n], labels[n] = self.string_to_example(strings[n])
        return [enc, labels]

    def examples_generator(self, strings):
        assert len(strings) >= 1
        while True:
            ind = 0
            while ind < len(strings):
                yield self.strings_to_examples(strings[ind : ind + self.batch_size])

                ind += self.batch_size

    def save(self, path):
        """
        Write a Preprocessor object to a .JSON config
        """
        config = {
            "char_rev": self.char_rev,
            "char_dict": self.char_dict,
            "max_example_len": self.max_example_len,
        }
        with open(os.path.join(path, "ecle_config.json"), "w") as f:
            json.dump(config, f)

    def _load(self, path):
        """
        Load a Preprocessor object from disk
        """
        with open(path, "rb") as f:
            result = json.load(f)
        for key, value in result.items():
            setattr(self, key, value)
