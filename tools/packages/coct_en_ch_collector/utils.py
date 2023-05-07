import re
import pangu
from bs4 import BeautifulSoup


def process_en_sent(text):
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()
    tokens = text.split(' ')
    new_text = ''
    in_double_quotes = False
    in_single_quotes = False
    skip_next = False
    for i, token in enumerate(tokens):
        if skip_next:
            skip_next = False
            continue
        if token == ',':
            new_text += token
        elif token == '.':
            new_text += token
        elif token == '?':
            new_text += token
        elif token == '??':
            new_text += token
        elif token == '!':
            new_text += token
        elif token == '!!':
            new_text += token
        elif token == '?!':
            new_text += token
        elif token == '!?':
            new_text += token
        elif token == ';':
            new_text += token
        elif token == ':':
            new_text += token
        # elif token == '/':
        #     new_text += token
        elif token == "s'":
            new_text += token
        elif token == '"':
            if in_double_quotes:
                new_text += token
                in_double_quotes = False
            else:
                new_text += ' '
                new_text += token
                next_token = get_from_list(tokens, i + 1)
                if next_token is not None:
                    new_text += next_token
                    skip_next = True
                in_double_quotes = True
        elif token == "'":
            if in_single_quotes:
                new_text += token
                in_single_quotes = False
            else:
                new_text += ' '
                new_text += token
                next_token = get_from_list(tokens, i + 1)
                if next_token is not None:
                    new_text += next_token
                    skip_next = True
                in_single_quotes = True
        elif token.startswith("'"):  # "This 's"
            new_text += token
        elif token == ")":
            new_text += token
        elif token == "]":
            new_text += token
        elif token == "(" or token == "[":
            new_text += ' '
            new_text += token
            next_token = get_from_list(tokens, i + 1)
            if next_token is not None:
                new_text += next_token
                skip_next = True
        else:
            new_text += ' '
            new_text += token

    return new_text.strip()


def get_from_list(list, index):
    try:
        return list[index]
    except IndexError:
        return None


CJK = r'\u2e80-\u2eff\u2f00-\u2fdf\u3040-\u309f\u30a0-\u30fa\u30fc-\u30ff\u3100-\u312f\u3200-\u32ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u4E00-\u9FFF\u3400-\u4DBF\uF900-\uFAFF\u2E80-\u2EFF\u31C0-\u31EF\uFE30-\uFE4F\u3000-\u303F\uFF01-\uFF60'
# CJK = r'\u4E00-\u9FFF\u3400-\u4DBF\u20000-\u2A6DF\u2A700-\u2B73F\u2B740-\u2B81F\u2B820-\u2CEAF\u2CEB0-\u2EBEF\uF900-\uFAFF\u2E80-\u2EFF\u31C0-\u31EF\uFE30-\uFE4F\u3000-\u303F\uFF01-\uFF60\u3200-\u32FF\u3300-\u33FF'
SPACES_AFTER_CJK = re.compile('([{CJK}]) +'.format(CJK=CJK))
SPACES_BEFORE_CJK = re.compile(' +([{CJK}])'.format(CJK=CJK))


def process_ch_sent(text):
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()
    text = SPACES_AFTER_CJK.sub(r'\1', text)
    text = SPACES_BEFORE_CJK.sub(r'\1', text)
    text = pangu.spacing_text(text)
    return text
