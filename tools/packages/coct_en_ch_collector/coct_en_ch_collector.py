import os
import argparse
import requests
import json
import traceback
from tqdm.auto import tqdm

import sys
from pathlib import Path
import importlib.util


def import_relative_file(module_name, relative_path):
    file_path = Path(__file__).parent / relative_path
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if not spec:
        raise Exception(f"Can't find file {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    if not spec.loader:
        raise Exception(f"Can't find file {file_path}")
    spec.loader.exec_module(module)
    return module


utils = import_relative_file("utils", "utils.py")
process_en_sent = utils.process_en_sent
process_ch_sent = utils.process_ch_sent

file_dir = os.path.dirname(os.path.abspath(__file__))
word_list_file_path = os.path.join(file_dir, 'word_list.txt')
output_file_path = os.path.join(file_dir, 'output.jsonl')
log_file_path = os.path.join(file_dir, 'log.txt')



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
        sents.append({
            'en': process_en_sent(sent['Esent']),
            'ch': process_ch_sent(sent['Csent']),
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


def main(argv=sys.argv[1:]):
    parser = argparse.ArgumentParser(
        prog='coct_en_ch_collector',
        # description='',
    )

    parser.add_argument(
        '--page_limit', type=int, default=10,
    )

    args = parser.parse_args(argv)

    print('Starting with page_limit:', args.page_limit)

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
            sample_sentences = get_sample_sentences(
                keyword,
                page_limit=args.page_limit
            )
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


if __name__ == "__main__":
    main()
