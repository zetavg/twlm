#!/usr/bin/env python

import argparse
import sys
import json
from datasets import Dataset


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

    with open('packages/coct_en_ch_collector/output.jsonl', 'r') as f:
        data = [json.loads(line) for line in f]
        print("data lines:", len(data))
        data_dict = {d['en']: d['ch'] for d in data}
        data_d = [{'en': k, 'ch': v }for k, v in data_dict.items()]
        print("deduped data lines:", len(data_d))
        ds = Dataset.from_list(data_d)
        print("dataset size:", len(ds))
        ds.push_to_hub(args.name)

    print(f"Done.")


if __name__ == "__main__":
    main()
