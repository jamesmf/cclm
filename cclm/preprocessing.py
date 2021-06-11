from tqdm import tqdm
import numpy as np
from tokenizers import Encoding, Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Lowercase, NFD, StripAccents, Sequence
from typing import List, Union
from datasets.arrow_dataset import Dataset
import json
import os
import re
from collections import Counter
from typing import Dict, List, Iterable, Optional

PUNCT = "~!@#$%^&*()-+=_.,?\":;'"


class Preprocessor:
    def __init__(
        self,
        load_from: str = None,
        vocab_size: int = 10000,
        max_example_len: int = 128,
        batch_size: int = 16,
        downsample_factor: int = 1,
        add_cls: bool = True,
    ):
        self.char_dict: Dict[str, int] = {}
        self.char_rev: Dict[int, str] = {}
        self.token_dict: Dict[str, int] = {}
        self.token_rev: Dict[str, int] = {}
        self.vocab_size = vocab_size
        self.max_example_len = max_example_len
        self.batch_size = batch_size
        self.add_cls = True
        # some default values
        self.mask_token_ind = 1
        self.unk_token_ind = 2
        self.cls_token_ind = 3

        self.downsample_factor = downsample_factor
        self.tokenizer_is_fit = False
        self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = Whitespace()
        self.tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])
        self.tok_trainer = BpeTrainer(
            special_tokens=["[UNK]", "[MASK]", "[CLS]"], vocab_size=self.vocab_size
        )
        if load_from:
            self._load(load_from)

    def fit(
        self,
        data: Union[List[str], Dataset],
        min_char_count: int = 1,
        skip_tokenizer: bool = False,
        progbar: bool = True,
    ):
        """
        Create a character-level dictionary based on a list of strings and train
        the tokenizer on the dataset. Training the tokenizer allows for using it
        in the language modeling objective
        """
        is_dataset = isinstance(data, Dataset)
        if not self.tokenizer_is_fit and not skip_tokenizer:
            print("fitting tokenizer")
            self.tokenizer.train_from_iterator(data, trainer=self.tok_trainer)
            self.tokenizer_is_fit = True
        char_counter: Counter = Counter()
        token_counter: Counter = Counter()
        iterator_: Iterable = data
        if progbar:
            iterator_ = tqdm(data)
        print("fitting char_dict")
        for example in iterator_:
            if is_dataset:
                example = example["text"]
            chars = Counter(example)
            for char, char_count in chars.items():
                try:
                    char_counter[char] += char_count
                except KeyError:
                    char_counter[char] = char_count

        counts = [k for k, v in char_counter.items() if v >= min_char_count]

        self.char_rev = {
            0: "",
            self.mask_token_ind: "_",
            self.cls_token_ind: "[CLS]",
            self.unk_token_ind: "",
        }

        for c in sorted(counts):
            n = len(self.char_rev)
            self.char_rev[n] = c
            self.char_dict[c] = n

    def tokenize(self, string_to_tokenize: str) -> Encoding:
        string_to_tokenize = string_to_tokenize
        return self.tokenizer.encode(string_to_tokenize)

    def string_to_array(
        self, string_in: str, length: Optional[int] = None, use_max_len: bool = True
    ):
        """Turn a string into an array[int]. If length specified, will
        truncate to that length. Otherwise, the function will infer
        the proper length. If the Preprocessor has .add_cls == True,
        then it will also add an int to the end representing [CLS]

        Args:
            string_in (str): string to vectorize
            length (Optional[int], optional): specific length of the output. Defaults to None.
            use_max_len (bool): whether to enforce a Preprocessor.max_example_len as maximum

        Returns:
            [type]: [description]
        """
        len_str = len(string_in)
        specific_length = True
        clip_char = False
        if length is None:
            specific_length = False
            length = self.infer_length(string_in, use_max_len)

        # we may need to clip an extra character to make room for [CLS]
        if self.add_cls and (len_str + 1 * self.add_cls) >= (length):
            clip_char = True
        # truncate
        s = string_in[: (length - 1 * clip_char)]
        # map char -> int and left-zero-pad
        mapped = np.zeros((length))
        for n, char in enumerate(s):
            try:
                mapped[n] = self.char_dict[char]
            except KeyError:
                pass

        # if we're adding the CLS token,
        if self.add_cls:
            mapped[n + 1] = self.cls_token_ind

        return mapped

    def infer_length(self, inp: str, use_max_len=True) -> int:
        """infer the length for string_to_array. If use_max_len, honor the maximum length
        from self.max_example_len. Otherwise, just take into account the necessary
        downsample_factor and add_cls

        Args:
            inp (str): input string
            use_max_len (bool, optional): Whether to honor self.max_example_len. Defaults to True.

        Returns:
            int: [description]
        """
        l = len(inp) + 1 * self.add_cls
        # possibly enforce max len, possibly truncate to a multiple of downsample_factor
        if use_max_len and l >= self.max_example_len:
            return l - (l % self.downsample_factor)
        # otherwise just enforce downsample_factor
        mod = l % self.downsample_factor
        if mod > 0:
            l += self.downsample_factor - mod
        return l

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
        for key, value in list(self.char_rev.items()):
            self.char_rev[int(key)] = value

    @property
    def n_chars(self):
        return np.max(list(self.char_dict.values())) + 1
