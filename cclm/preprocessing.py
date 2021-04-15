from tqdm import tqdm
import numpy as np
from tokenizers import BertWordPieceTokenizer, Encoding, Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Lowercase, NFD, StripAccents, Sequence
from typing import List
import json
import os
import re
from collections import Counter
from typing import Dict, List, Iterable

PUNCT = "~!@#$%^&*()-+=_.,?\":;'"


class MLMPreprocessor:
    def __init__(
        self,
        load_from: str = None,
        vocab_size: int = 10000,
        max_example_len: int = 128,
        batch_size: int = 16,
        num_stopwords: int = 250,
        mask_output_len: int = 4,
    ):
        self.char_dict: Dict[str, int] = {}
        self.char_rev: Dict[int, str] = {}
        self.token_dict: Dict[str, int] = {}
        self.token_rev: Dict[str, int] = {}
        self.vocab_size = vocab_size
        self.max_example_len = max_example_len
        self.batch_size = batch_size
        self.num_stopwords = num_stopwords
        self.mask_output_len = mask_output_len
        self.tokenizer_fit = False
        self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = Whitespace()
        self.tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])
        self.tok_trainer = BpeTrainer(
            special_tokens=["[UNK]", "[MASK]"], vocab_size=self.vocab_size
        )
        if load_from:
            self._load(load_from)

    def fit(self, data: List[str], min_char_freq: int = 1, progbar: bool = True):
        """
        Create a character-level dictionary based on a list of strings and train
        the tokenizer on the dataset. Training the tokenizer allows for using it
        in the language modeling objective
        """
        if not self.tokenizer_fit:
            self.tokenizer.train_from_iterator(data, trainer=self.tok_trainer)
        char_counter: Counter = Counter()
        token_counter: Counter = Counter()
        iterator_: Iterable = data
        if progbar:
            iterator_ = tqdm(data)
        for example in iterator_:
            chars = Counter(example)
            for char, char_count in chars.items():
                try:
                    char_counter[char] += char_count
                except KeyError:
                    char_counter[char] = char_count

        counts = [k for k, v in char_counter.items() if v >= min_char_freq]
        self.char_rev = {0: "", 1: "?", 2: "?", 3: ""}

        for c in sorted(counts):
            n = len(self.char_rev)
            self.char_rev[n] = c
            self.char_dict[c] = n

    def tokenize(self, string_to_tokenize: str) -> Encoding:
        string_to_tokenize = string_to_tokenize
        return self.tokenizer.encode(string_to_tokenize)

    def string_to_array(self, string_in: str, length: int, padding_pre: bool = False):
        # truncate
        if padding_pre:
            s = string_in[-length:]
        else:
            s = string_in[:length]
        # map char -> int and left-zero-pad
        mapped = np.ones((len(s)))
        for n, char in enumerate(s):
            try:
                mapped[n] = self.char_dict[char]
            except KeyError:
                pass
        # mapped = [self.char_dict.get(x, 1) for x in s]
        if padding_pre:
            r = np.pad(mapped, (length - len(s), 0), "constant", constant_values=(0, 0))
        else:
            r = np.pad(mapped, (0, length - len(s)), "constant", constant_values=(0, 0))
        return r

    def save(self, path: str):
        """
        Write a Preprocessor object to a .JSON config
        """
        config = {
            "char_rev": self.char_rev,
            "char_dict": self.char_dict,
            "max_example_len": self.max_example_len,
        }
        with open(os.path.join(path, "cclm_config.json"), "w") as f:
            json.dump(config, f)

    def _load(self, path: str):
        """
        Load a Preprocessor object from disk
        """
        with open(path, "rb") as f:
            result = json.load(f)
        for key, value in result.items():
            setattr(self, key, value)
