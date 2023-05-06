import os
import re
import pangu
import requests
import json
import traceback
from bs4 import BeautifulSoup
from tqdm.auto import tqdm

file_dir = os.path.dirname(os.path.abspath(__file__))
word_list_file_path = os.path.join(file_dir, 'word_list.txt')
output_file_path = os.path.join(file_dir, 'output.jsonl')
log_file_path = os.path.join(file_dir, 'log.txt')


def process_en_sent(text):
    tokens = text.split(' ')
    new_text = ''
    in_double_quotes = False
    in_single_quotes = False
    skip_next = False
    for i, token in enumerate(tokens):
        if skip_next:
            skip_next = False
            continue
        if not new_text:
            new_text = token
        elif token == ',':
            new_text += token
        elif token == '.':
            new_text += token
        elif token == '!':
            new_text += token
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
    text = SPACES_AFTER_CJK.sub(r'\1', text)
    text = SPACES_BEFORE_CJK.sub(r'\1', text)
    text = pangu.spacing_text(text)
    return text


def get_sample_sentences(keyword, page_limit=100, from_page=0, progress=None):
    # print("from_page", from_page, "page_limit", page_limit)
    if page_limit and from_page >= page_limit:
        return []

    api_url = 'https://coct.naer.edu.tw/bc/getSampleSentences/'
    params = {
        'CorpusName': 'TWP',
        'qword': keyword,
        'current_page': from_page
    }
    response = requests.get(
        api_url, params=params,
    )

    json = response.json()
    sents = []
    for sent in json['sents']:
        c_soup = BeautifulSoup(sent['Csent'], "html.parser")
        c_text = c_soup.get_text()
        c_text = process_ch_sent(c_text)
        e_soup = BeautifulSoup(sent['Esent'], "html.parser")
        e_text = e_soup.get_text()
        e_text = process_en_sent(e_text)
        sents.append({
            'en': e_text,
            'ch': c_text,
            'raw_en': sent['Esent'],
            'raw_ch': sent['Csent'],
        })

    if json['total_page'] > 1 and from_page == 0:
        progress = tqdm(total=min(page_limit, json['total_page']))

    if progress:
        progress.update(1)

    if json['current_page'] < json['total_page']:
        return sents + get_sample_sentences(
            keyword,
            page_limit=page_limit,
            from_page=json['current_page'] + 1,
            progress=progress,
        )
    else:
        return sents


with open(word_list_file_path, 'r') as f:
    word_list = f.read().splitlines()
print('word_list count:', len(word_list))

errored_words = []

f = open(output_file_path, 'w')
log_f = open(log_file_path, 'w')
for keyword in tqdm(word_list):
    log_f.write(keyword + '\n')
    tqdm.write(keyword)
    try:
        sample_sentences = get_sample_sentences(keyword)
        for sample_sentence in sample_sentences:
            f.write(json.dumps(sample_sentence, ensure_ascii=False) + '\n')
    except Exception as e:
        message = f"Error getting samples for word '{keyword}': {e} ({traceback.format_tb(e.__traceback__)})"
        log_f.write(message + '\n')
        tqdm.write(message)
        errored_words.append(keyword)
    f.flush()
    log_f.flush()

print("Done.")
if len(errored_words):
    print('errored_words:', json.dumps(errored_words, ensure_ascii=False))
