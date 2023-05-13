from typing import Union

import os
import re
import fire
import json
from transformers import AutoTokenizer
from tokenizers import Tokenizer as TokenizerFast
from datasets import Dataset, load_dataset, concatenate_datasets
from huggingface_hub import HfFileSystem
from tqdm import tqdm
from termcolor import colored
from textwrap import indent, dedent

from utils.config import Config
from utils.paths import Paths
from utils.load import load_tokenizer
from utils.formatting import (
    better_format_pairs_in_json_text,
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


def prepare_dataset(
    train_name: str,
    cfg: Union[str, None] = None,
    config_file_path: Union[str, None] = None,
    data_dir_path: Union[str, None] = None,
    do_not_save=False,
):
    paths = Paths(data_dir_path)
    if cfg and not config_file_path:
        config_file_path = paths.get_config_path(cfg)
    config = Config(config_file_path)
    training_config = config.get_training_config(train_name)
    dataset_config = training_config.dataset_config

    message = ''
    message += f"Preparing dataset '{dataset_config.dataset_name}' "
    message += f"using tokenizer '{config.tokenizer_name}' "
    message += f"with '{dataset_config.build_with}' "
    message += f", max_training_text_length={dataset_config.max_training_text_length} "
    message += "..."
    print(message)
    print()

    tokenizer = load_tokenizer(config, paths)

    datasets = []

    for build_type in dataset_config.build_with:
        if build_type == 'translations':
            ds = generate_translations_dataset(
                tokenizer, dataset_config,
                dataset_config.get_settings_for(build_type))
            datasets.append(ds)

        if build_type == 'alpaca':
            ds = generate_alpaca_dataset(
                tokenizer, dataset_config,
                dataset_config.get_settings_for(build_type))
            datasets.append(ds)

        else:
            raise Exception(
                f"Unknown dataset build method '{build_type}'. Check {dataset_config.config_file_path}")

    print("Preparing final dataset...")
    dataset = concatenate_datasets(datasets)
    print(f"Concatenated dataset contains {len(dataset)} rows.")

    dataset = dataset.filter(lambda x: x['input_ids'])
    dataset = dataset.shuffle()
    print(colored(
        f"Final dataset contains {len(dataset)} rows.",
        attrs=['bold']
    ))
    print()

    if not do_not_save:
        print('Saving dataset...')
        dataset_save_path = paths.get_dataset_path(dataset_config.dataset_name)
        dataset.save_to_disk(dataset_save_path)
        print(f"Dataset saved to {dataset_save_path}.")
        print()

    if config.push_outputs_to_hf:
        hf_dataset_name = \
            f"{config.hf_user_or_org_name}/{dataset_config.dataset_name}"
        hf_dataset_path = f"datasets/{hf_dataset_name}"

        if not do_not_save:
            print('Pushing dataset to HF Hub...')
            dataset.push_to_hub(hf_dataset_name, private=True)

        print("Updating dataset card...")
        dataset_card_frontmatter = {}
        dataset_card_content = dedent(f"""
        # {dataset_config.dataset_name}

        This dataset is a part of the `{config.project_name}` project.

        * Tokenizer: `{config.tokenizer_name}`
        * Built with: {', '.join(f"`{s}`" for s in dataset_config.build_with)}
        * Rows: `{len(dataset)}`
        * Max length: `{dataset_config.max_training_text_length}`
        * Full config:
          ```json
          {dataset_config.to_json()}
          ```
        """).strip()
        update_hf_readme(
            hf_dataset_path,
            dataset_card_content,
            dataset_card_frontmatter)
        print(colored(
            f"Dataset uploaded to https://huggingface.co/{hf_dataset_path}.",
            attrs=['bold']
        ))


def get_tokenize_data_fn(tokenizer, dataset_column, max_length, preview_length):
    def tokenize_data(data_point):
        batch_encoding = tokenizer(
            # See: https://huggingface.co/docs/transformers/main/en/main_classes/tokenizer#tokenizer
            data_point[dataset_column],
            max_length=max_length,
            truncation=True,
            padding=False,  # Handled by DataCollatorForSeq2Seq.
            return_tensors=None  # Handled by the trainer.
        )
        if isinstance(data_point[dataset_column], list):
            # is batched
            batch_encoding['labels'] = []
            batch_encoding['preview'] = []
            for i, source_text in enumerate(data_point[dataset_column]):
                batch_encoding['labels'].append(
                    batch_encoding['input_ids'][i].copy())
                preview_text = source_text
                if len(preview_text) > preview_length:
                    preview_text = preview_text[:preview_length] + ' [...]'
                batch_encoding['preview'].append(preview_text)
        else:
            # not batched
            batch_encoding["labels"] = batch_encoding["input_ids"].copy()
            preview_text = source_text
            if len(preview_text) > preview_length:
                preview_text = preview_text[:preview_length] + ' [...]'
            batch_encoding["preview"] = preview_text
        return batch_encoding
    return tokenize_data


def generate_translations_dataset(tokenizer, dataset_config, settings):
    source_ds_name = settings.get('source_dataset')
    assert source_ds_name, f"{dataset_config.get_config_level_str(['translations_settings', 'source_dataset'])} is missing in config {dataset_config.config_file_path}."

    lang_1_key = settings.get('lang_1_key')
    assert lang_1_key, f"{dataset_config.get_config_level_str(['translations_settings', 'lang_1_key'])} is missing in config {dataset_config.config_file_path}."
    lang_2_key = settings.get('lang_2_key')
    assert lang_2_key, f"{dataset_config.get_config_level_str(['translations_settings', 'lang_2_key'])} is missing in config {dataset_config.config_file_path}."
    templates = settings.get('templates')
    assert templates, f"{dataset_config.get_config_level_str(['translations_settings', 'templates'])} is missing in config {dataset_config.config_file_path}."

    rows_limit = settings.get('rows_limit')

    print(f"Loading translations dataset '{source_ds_name}'...")
    source_ds: Dataset = \
        load_dataset(source_ds_name)['train']  # type: ignore

    print('Processing translations dataset...')

    if rows_limit:
        print(f"Limiting to {rows_limit} rows.")
        source_ds = source_ds.select(range(rows_limit))

    def get_translations_text(batch):
        output = {'text': []}

        for lang_1_text, lang_2_text in zip(
                batch[lang_1_key], batch[lang_2_key]):

            for template in templates:
                text = template.format(
                    lang_1=lang_1_text,
                    lang_2=lang_2_text,
                )
                output['text'].append(text.strip())

        return output

    ds = source_ds.map(
        get_translations_text,
        batched=True,
        remove_columns=list(source_ds.features.keys()))

    print('Tokenizing translations dataset...')

    ds = ds.map(
        get_tokenize_data_fn(
            tokenizer=tokenizer,
            dataset_column='text',
            max_length=dataset_config.max_training_text_length,
            preview_length=dataset_config.preview_length,
        ),
        remove_columns=['text'],
        batched=True,
        batch_size=512,
    )

    print(f"Translations dataset ok. Has {len(ds)} items.")
    print()

    return ds


def generate_alpaca_dataset(tokenizer, dataset_config, settings):
    source_ds_name = settings.get('source_dataset')
    assert source_ds_name, f"{dataset_config.get_config_level_str(['translations_settings', 'source_dataset'])} is missing in config {dataset_config.config_file_path}."

    rows_limit = settings.get('rows_limit')
    template = settings.get('template')

    print(f"Loading alpaca dataset '{source_ds_name}'...")
    source_ds: Dataset = \
        load_dataset(source_ds_name)['train']  # type: ignore

    print('Processing alpaca dataset...')

    if rows_limit:
        print(f"Limiting to {rows_limit} rows.")
        source_ds = source_ds.select(range(rows_limit))

    def get_alpaca_text(batch):
        batch_output = {'text': []}

        for instruction, input, output in zip(
                batch['instruction'], batch['input'], batch['output']):

            if template == 'short':
                if input:
                    text = dedent(f"""
                        ### Instruction:
                        {instruction}

                        ### Input:
                        {input}

                        ### Response:
                        {output}
                    """).strip()
                else:
                    text = dedent(f"""
                        ### Instruction:
                        {instruction}

                        ### Response:
                        {output}
                    """).strip()
            else:
                if input:
                    text = dedent(f"""
                        Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

                        ### Instruction:
                        {instruction}

                        ### Input:
                        {input}

                        ### Response:
                        {output}
                    """).strip()
                else:
                    text = dedent(f"""
                        Below is an instruction that describes a task. Write a response that appropriately completes the request.

                        ### Instruction:
                        {instruction}

                        ### Response:
                        {output}
                    """).strip()
            batch_output['text'].append(text.strip())
        return batch_output

    ds = source_ds.map(
        get_alpaca_text,
        batched=True,
        remove_columns=list(source_ds.features.keys()))

    print('Tokenizing alpaca dataset...')

    ds = ds.map(
        get_tokenize_data_fn(
            tokenizer=tokenizer,
            dataset_column='text',
            max_length=dataset_config.max_training_text_length,
            preview_length=dataset_config.preview_length,
        ),
        remove_columns=['text'],
        batched=True,
        batch_size=512,
    )

    print(f"Alpaca dataset ok. Has {len(ds)} items.")
    print()

    return ds


if __name__ == "__main__":
    fire.Fire(prepare_dataset)
