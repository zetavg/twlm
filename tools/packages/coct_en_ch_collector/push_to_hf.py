#!/usr/bin/env python

import os
import argparse
import sys
import json
from datasets import Dataset

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
output_file_path = os.path.join(file_dir, 'output.jsonl')


def main(argv=sys.argv[1:]):
    parser = argparse.ArgumentParser(
        prog='push_to_hf',
        # description='',
    )

    parser.add_argument(
        'name', nargs='?',
        # help=""
    )

    args = parser.parse_args(argv)

    if not args.name:
        print("No name provided. Exiting. For more information, run with --help.")
        sys.exit(1)

    print(f"Pushing to HF Hub as '{args.name}'...")

    with open(output_file_path, 'r') as f:
        data = [json.loads(line) for line in f]
        print("data lines:", len(data))
        data_dict = {
            process_en_sent(d['raw_en']) if d.get('raw_en') else d['en']:
            process_ch_sent(d['raw_ch']) if d.get('raw_ch') else d['ch']
            for d in data}
        data_d = [{'en': k, 'ch': v} for k, v in data_dict.items()]
        print("deduped data lines:", len(data_d))
        ds = Dataset.from_list(data_d)
        print("dataset size:", len(ds))
        ds.push_to_hub(args.name)

    print(f"Done.")


if __name__ == "__main__":
    main()
