from difflib import SequenceMatcher


def longest_common_substring(s1, s2):
    match = SequenceMatcher(None, s1, s2).find_longest_match(
        0, len(s1), 0, len(s2))
    return s1[match.a: match.a + match.size]


def remove_longest_common_substring(s1, s2, min_length=4):
    lcs = longest_common_substring(s1, s2)
    if len(lcs) < min_length:
        return s1
    return s1.replace(lcs, '', 1)


def remove_common_substring(s1, s2, min_length=4, passes=5):
    for _ in range(passes):
        s1 = remove_longest_common_substring(s1, s2, min_length)
    return s1
