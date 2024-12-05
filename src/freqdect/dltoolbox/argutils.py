from argparse import ArgumentTypeError
from typing import List, Tuple


def parse_image_size(values: List[int]) -> Tuple[int, int]:
    if len(values) == 1:
        return values[0], values[0]
    elif len(values) == 2:
        return values[0], values[1]
    else:
        raise ArgumentTypeError("failed to parse image size, expected one or two integers")


def parse_dataset_size(values: List[int]) -> Tuple[int, int, int]:
    if len(values) == 1:
        return values[0], 0, 0
    elif len(values) == 2:
        return values[0], values[1], 0
    elif len(values) == 3:
        return values[0], values[1], values[2]
    else:
        raise ArgumentTypeError("failed to parse dataset size, expected one, two, three integers")
