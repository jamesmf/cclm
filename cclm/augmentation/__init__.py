import typing
from typing import List, Optional, Tuple, Dict, Union, Callable
import numpy as np

CallableType = Callable[[str], str]
TransformType = Dict[str, Union[CallableType, float]]
TransformationDictType = Dict[str, TransformType]
TransformationListType = List[Tuple[CallableType, float]]

CONST_FUNCTION = "function"
CONST_PROBABILITY = "probability"


def lower(x: str) -> str:
    return x.lower()


def upper(x: str) -> str:
    return x.upper()


def title_first(x: str) -> str:
    return x[0].upper() + x[1:]


class Augmentor:
    """
    Apply a list of transformations as text augmentation
    """

    default_transformations: TransformationDictType = {
        "lower": {
            CONST_FUNCTION: lower,
            CONST_PROBABILITY: 0.02,
        },
        "upper": {CONST_FUNCTION: upper, CONST_PROBABILITY: 0.005},
        "title_first": {CONST_FUNCTION: title_first, CONST_PROBABILITY: 0.01},
    }

    def __init__(self, transformations: Optional[TransformationDictType] = None):
        if transformations is not None:
            self.transformation_dict = typing.cast(
                TransformationDictType, transformations
            )
        else:
            self.transformation_dict = self.default_transformations
        self.transformations = self.compile()

    def compile(self):
        transform_list: TransformationListType = []
        for transform_name, transform in sorted(self.transformation_dict.items()):
            transform_list.append(
                (transform[CONST_FUNCTION], transform[CONST_PROBABILITY])
            )
        return transform_list

    def transform(self, x: str) -> str:
        for transform, probability in self.transformations:
            if np.random.rand() < probability:
                x = transform(x)
        return x