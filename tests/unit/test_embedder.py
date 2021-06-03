import pytest
import os
from cclm.models import Embedder


def test_persist_embedder(tmp_path):
    emb_dir = tmp_path / ".embedder_test"
    emb = Embedder(max_len=5)
    emb.save(emb_dir)
    print(os.listdir(emb_dir))

    emb2 = Embedder(load_from=emb_dir)
    assert emb2.max_len == 5, "did not properly persist attribute max_len"
