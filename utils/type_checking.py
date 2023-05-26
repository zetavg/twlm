from typing import Any, List


def assert_list_of_strings(lst: Any, name='Value') -> List[str]:
    is_list_of_strings = (
        isinstance(lst, list)
        and all(isinstance(item, str) for item in lst)
    )
    assert is_list_of_strings, f"{name} must be a list contain only strings"
    return lst
