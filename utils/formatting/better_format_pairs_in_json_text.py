import re


def better_format_pairs_in_json_text(text):
    return re.sub(
        r'\[[\n ]*([0-9]+),[\n ]*([0-9]+)[\n ]*\]',
        r'[\1, \2]',
        text
    )
