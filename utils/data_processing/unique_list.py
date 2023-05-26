from typing import List, TypeVar

T = TypeVar("T")


def unique_list(list: List[T]) -> List[T]:
    """
    Return a list with duplicate elements removed, and the order preserved.
    """
    seen = set()
    return [x for x in list if not (x in seen or seen.add(x))]
