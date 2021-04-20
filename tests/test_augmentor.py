import pytest
from cclm.augmentation import Augmentor


def test_augmentor():
    aug = Augmentor()


def test_title_first():
    x = "a strinG"
    config = {"title_first": Augmentor.default_transformations["title_first"]}
    config["title_first"]["probability"] = 1.0
    aug = Augmentor(config)
    print(config)
    print(aug.transformations)
    assert aug.transform(x) == "A strinG", "failed title-casing first letter"
