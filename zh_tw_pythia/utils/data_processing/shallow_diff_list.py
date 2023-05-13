from typing import Any, List, TypedDict, TypeVar

T = TypeVar("T")

DiffResults = TypedDict(
    'DiffResults', {'added': List[Any], 'removed': List[Any]})


def shallow_diff_list(list1: List[T], list2: List[T]) -> DiffResults:
    set1 = set(list1)
    set2 = set(list2)

    added = set1.difference(set2)
    removed = set2.difference(set1)

    return {
        'added': list(added),
        'removed': list(removed),
    }
