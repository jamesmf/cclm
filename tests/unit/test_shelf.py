import pytest
import os
from cclm.shelf import Shelf
from cclm.preprocessing import Preprocessor


def test_download(tmp_path):
    cache_dir = tmp_path / ".shelf_test"
    shelf = Shelf()
    identifier = "en_wiki_clm_1"
    item_type = "preprocessor"
    shelf.fetch(identifier, item_type, cache_dir=cache_dir, tag="en_wiki_clm_1")
    print(os.listdir(os.path.join(cache_dir, identifier, item_type)))
    prep = Preprocessor(
        load_from=os.path.join(cache_dir, identifier, item_type, f"cclm_config.json")
    )
    assert (
        len(prep.char_dict) > 5
    ), "did not create en_wiki_clm_1/preprocessor when shelf was downloading model en_wiki_clm_1"
