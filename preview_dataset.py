import pdb
from typing import Any, Union, List, Dict

import os
import re
import fire
import json
import wandb
import traceback
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForSeq2Seq,
    TrainerCallback
)
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from tokenizers import Tokenizer as TokenizerFast
from datasets import Dataset, load_dataset, concatenate_datasets
from huggingface_hub import HfFileSystem
from tqdm import tqdm
from termcolor import colored
from textwrap import indent, dedent

from utils.config import Config
from utils.paths import Paths, project_dir
from utils.get_training_config_values import get_training_config_values
from utils.load import load_tokenizer, load_dataset
from utils.formatting import (
    better_format_pairs_in_json_text,
    comparing_lists,
    get_human_timestamp,
    human_short_number as hs_number,
    truncate_string_by_lines,
)
from utils.data_processing import (
    shallow_diff_dict,
    unique_list,
)
from utils.type_checking import (
    assert_list_of_strings,
)
from utils.tokenize_splits_preview import tokenize_splits_preview
from utils.update_hf_readme import update_hf_readme


def preview_dataset(
    train_name: str,
    split: str = 'train',
    range_: str = '0,100',
    only_preview: bool = False,
    cfg: Union[str, None] = None,
    config_file_path: Union[str, None] = None,
    data_dir_path: Union[str, None] = None,
):
    r = range_.split(',') if isinstance(range_, str) else range_
    start = int(r[0])
    end = int(r[1])

    paths = Paths(data_dir_path)
    if cfg and not config_file_path:
        config_file_path = paths.get_config_path(cfg)
    config = Config(config_file_path)

    training_config = config.get_training_config(train_name)

    model_name, base_model_name, tokenizer_name, base_on_model_name, dataset_name, peft_type = map(
        get_training_config_values(config, training_config).get,
        ('model_name', 'base_model_name', 'tokenizer_name', 'base_on_model_name', 'dataset_name', 'peft_type'))

    tokenizer = load_tokenizer(config, paths)

    dataset = load_dataset(config, paths, dataset_name)
    dataset = dataset[split]
    print(f"Rows count: {len(dataset)}")
    dataset = dataset.select(range(start, end))


    for i, row in enumerate(dataset):
        preview = row['preview'].replace('\n', '\\n')
        print(f"Row {i}: (l: {row.get('length')}) '{preview}'")
        if only_preview:
            print()
            continue
        print(comparing_lists(
            [
                [tokenizer.decode([i]) for i in row['input_ids']],
                [tokenizer.decode([i]) if i >= 0 else '' for i in row['labels']],
                row['input_ids'],
                row['attention_mask'],
                row['labels'],
            ],
            labels=['Inputs', 'Labels', 'input_ids', 'attention_mask', 'labels'],
            colors=[None, None, 'dark_grey', 'dark_grey', 'dark_grey'],
            add_blank_line=False,
        ))

if __name__ == "__main__":
    fire.Fire(preview_dataset)
