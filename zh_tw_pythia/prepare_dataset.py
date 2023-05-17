from typing import Any, Union

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
    message += f", max_tokens_length={dataset_config.max_tokens_length} "
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

        elif build_type == 'wikipedia':
            ds = generate_wikipedia_dataset(
                tokenizer, dataset_config,
                dataset_config.get_settings_for(build_type))
            datasets.append(ds)

        elif build_type == 'sharegpt':
            ds = generate_sharegpt_dataset(
                tokenizer, dataset_config,
                dataset_config.get_settings_for(build_type))
            datasets.append(ds)

        elif build_type == 'alpaca':
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
        * Max length: `{dataset_config.max_tokens_length}`
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
            max_length=dataset_config.max_tokens_length,
            preview_length=dataset_config.preview_length,
        ),
        remove_columns=['text'],
        batched=True,
        batch_size=512,
    )

    print(f"Translations dataset ok. Has {len(ds)} items.")
    print()

    return ds


def generate_wikipedia_dataset(tokenizer, dataset_config, settings):
    source_ds_name = settings.get('source_dataset')
    assert source_ds_name, f"{dataset_config.get_config_level_str(['wikipedia_settings', 'source_dataset'])} is missing in config {dataset_config.config_file_path}."

    rows_limit = settings.get('rows_limit')
    exclude = settings.get('exclude')

    print(f"Loading wikipedia dataset '{source_ds_name}'...")
    source_ds: Dataset = \
        load_dataset(source_ds_name)['train']  # type: ignore
    print(f"Dataset has {len(source_ds)} rows.")

    print('Processing wikipedia dataset...')

    if exclude:
        print(f"Filtering out rows with {len(exclude)} exclusion rules...")

        def filter_out_exclusions(data_point):
            for exc in exclude:
                if exc.get('content_length_longer_than'):
                    if len(data_point['markdown']) < exc['content_length_longer_than']:
                        return False
                    continue
                text = data_point[exc['in']]
                if exc.get('in_range'):
                    text = text[exc['in_range'][0]:exc['in_range'][1]]
                if exc['match'] in text:
                    return False
            return True

        source_ds = source_ds.filter(filter_out_exclusions)
        print(f"Dataset has {len(source_ds)} rows after filtered.")

    if rows_limit:
        print(f"Limiting to {rows_limit} rows.")
        source_ds = source_ds.select(range(rows_limit))

    print('Tokenizing wikipedia dataset...')

    ds = source_ds.map(
        get_tokenize_data_fn(
            tokenizer=tokenizer,
            dataset_column='markdown',
            max_length=dataset_config.max_tokens_length,
            preview_length=dataset_config.preview_length,
        ),
        remove_columns=list(source_ds.features.keys()),
        batched=True,
        batch_size=512,
    )

    print(f"Wikipedia dataset ok. Has {len(ds)} items.")
    print()

    return ds


def generate_sharegpt_dataset(tokenizer, dataset_config, settings):
    source_ds_name = settings.get('source_dataset')
    assert source_ds_name, f"{dataset_config.get_config_level_str(['sharepgt_settings', 'source_dataset'])} is missing in config {dataset_config.config_file_path}."

    rows_limit = settings.get('rows_limit')
    languages = settings.get('languages')
    train_on_inputs = settings.get('train_on_inputs')

    print(f"Loading ShareGPT dataset '{source_ds_name}'...")
    source_ds: Dataset = \
        load_dataset(source_ds_name)['train']  # type: ignore

    print('Processing ShareGPT dataset...')

    source_ds = source_ds.shuffle()
    if not rows_limit or rows_limit > len(source_ds):
        rows_limit = len(source_ds)
    lang_limits = {}
    if languages:
        lang_limits = {
            list(lang.keys())[0]: list(lang.values())[0]  # type: ignore
            for lang in languages if isinstance(lang, dict)}
    lang_counts = {}
    languages = [
        la if isinstance(la, str) else list(la.keys())[0]
        for la in languages]

    def data_generator():
        progress_bar = tqdm(total=rows_limit)
        rows_yield = 0
        i = 0
        while rows_yield < rows_limit and i <= len(source_ds):
            d = source_ds[i]
            if languages:
                if d['lang'] not in languages:
                    i += 1
                    continue
            if lang_limits.get(d['lang']) and lang_counts.get(d['lang']) and lang_counts.get(d['lang']) > lang_limits.get(d['lang']):  # type: ignore
                i += 1
                continue
            if not lang_counts.get(d['lang']):
                lang_counts[d['lang']] = 1
            else:
                lang_counts[d['lang']] += 1
            yield d
            rows_yield += 1
            i += 1
            progress_bar.update(1)

    ds: Dataset = Dataset.from_generator(data_generator)  # type: ignore
    print(f"Dataset has {len(ds)} rows after filtered.")
    print("Languages:", json.dumps(lang_counts, indent=2))

    print('Tokenizing ShareGPT dataset...')

    def tokenize(text):
        result = tokenizer(
            text,
            max_length=dataset_config.max_tokens_length,
            truncation=True,
            padding=False,  # Handled by DataCollatorForSeq2Seq.
            return_tensors=None  # Handled by the trainer.
        )
        return result

    def tokenize_message(text, f) -> Any:
        if f == 'human':
            result_1 = tokenize('### Human:\n')
            result_2 = tokenize(f"{text}\n\n")
            output = {
                'input_ids': result_1['input_ids'].copy(),
                'labels': result_1['input_ids'].copy(),
                'attention_mask': result_1['attention_mask'].copy(),
            }

            output['input_ids'] += result_2['input_ids'].copy()
            output['attention_mask'] += result_2['attention_mask'].copy()
            if train_on_inputs:
                output['labels'] += result_2['input_ids'].copy()
            else:
                output['labels'] += [-100] * len(result_2['input_ids'])

            return output

        elif f == 'gpt':
            output = tokenize(f"### AI:\n{text}\n\n")
            output['labels'] = output['input_ids'].copy()
            return output

        else:
            print(f"WARNING: unknown 'from' value: '{f}' ('{text}')")
            return None

    def tokenize_data(data_point):
        output = {
            'input_ids': [],
            'attention_mask': [],
            'labels': [],
        }

        ending_result = tokenize('### Human:\n')

        last_is_input = False
        for c in data_point['conversations']:
            message = c['opencc_converted_markdown'] or c['markdown']
            result = tokenize_message(message, c['from'])
            if not result:
                continue

            if (len(output['input_ids']) + len(result['input_ids']) + len(ending_result['input_ids'])) > dataset_config.max_tokens_length:
                break

            if c['from'] == 'human':
                last_is_input = True
            else:
                last_is_input = False

            output['input_ids'] += result['input_ids']
            output['attention_mask'] += result['attention_mask']
            output['labels'] += result['labels']

        if not last_is_input:
            output['input_ids'] += ending_result['input_ids']
            output['attention_mask'] += ending_result['attention_mask']
            output['labels'] += ending_result['input_ids']

        if len(output['input_ids']) > dataset_config.max_tokens_length:
            output['input_ids'] = output['input_ids'][:dataset_config.max_tokens_length]
            output['attention_mask'] = output['attention_mask'][:dataset_config.max_tokens_length]
            output['labels'] = output['labels'][:dataset_config.max_tokens_length]

        preview_length = dataset_config.preview_length
        preview_text = f"[{data_point['id']}] " + tokenizer.decode(output['input_ids'])
        if len(preview_text) > preview_length:
            preview_text = preview_text[:preview_length] + ' [...]'
        output["preview"] = preview_text

        return output

    ds = ds.map(
        tokenize_data,
        remove_columns=list(source_ds.features.keys())
    )

    print(f"ShareGPT dataset ok. Has {len(ds)} rows.")
    print()

    return ds


def generate_alpaca_dataset(tokenizer, dataset_config, settings):
    source_ds_name = settings.get('source_dataset')
    assert source_ds_name, f"{dataset_config.get_config_level_str(['translations_settings', 'source_dataset'])} is missing in config {dataset_config.config_file_path}."

    rows_limit = settings.get('rows_limit')
    template = settings.get('template')
    train_on_inputs = settings.get('train_on_inputs')

    print(f"Loading alpaca dataset '{source_ds_name}'...")
    source_ds: Dataset = \
        load_dataset(source_ds_name)['train']  # type: ignore

    print('Processing alpaca dataset...')

    if rows_limit:
        print(f"Limiting to {rows_limit} rows.")
        source_ds = source_ds.select(range(rows_limit))

    def get_alpaca_text(batch):
        batch_output = {'prompt': [], 'completion': []}

        for instruction, input, output in zip(
                batch['instruction'], batch['input'], batch['output']):

            if template == 'short':
                if input:
                    prompt = dedent(f"""
                        ### Instruction:
                        {instruction}

                        ### Input:
                        {input}

                        ### Response:
                    """).strip()
                else:
                    prompt = dedent(f"""
                        ### Instruction:
                        {instruction}

                        ### Response:
                    """).strip()
            else:
                if input:
                    prompt = dedent(f"""
                        Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

                        ### Instruction:
                        {instruction}

                        ### Input:
                        {input}

                        ### Response:
                    """).strip()
                else:
                    prompt = dedent(f"""
                        Below is an instruction that describes a task. Write a response that appropriately completes the request.

                        ### Instruction:
                        {instruction}

                        ### Response:
                    """).strip()
            completion = output
            batch_output['prompt'].append(prompt.strip() + '\n')
            batch_output['completion'].append(completion.strip())
        return batch_output

    ds = source_ds.map(
        get_alpaca_text,
        batched=True,
        remove_columns=list(source_ds.features.keys()))

    print('Tokenizing alpaca dataset...')

    def tokenize(text, add_eos_token=True):
        result = tokenizer(
            text,
            max_length=dataset_config.max_tokens_length,
            truncation=True,
            padding=False,  # Handled by DataCollatorForSeq2Seq.
            return_tensors=None  # Handled by the trainer.
        )
        if (
            add_eos_token
            and result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < dataset_config.max_tokens_length
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        return result

    def tokenize_data(data_point):
        text = data_point['prompt'] + data_point['completion']
        result = tokenize(text)

        if train_on_inputs:
            result["labels"] = result["input_ids"].copy()
        else:
            tokenized_prompt = tokenize(
                data_point['prompt'], add_eos_token=False)
            prompt_len = len(tokenized_prompt["input_ids"])

            labels = [-100] * prompt_len
            labels += result["input_ids"][prompt_len:]

            result["labels"] = labels

        preview_length = dataset_config.preview_length
        preview_text = text
        if len(preview_text) > preview_length:
            preview_text = preview_text[:preview_length] + ' [...]'
        result["preview"] = preview_text

        return result

    ds = ds.map(
        tokenize_data,
        remove_columns=['prompt', 'completion'],
    )

    print(f"Alpaca dataset ok. Has {len(ds)} items.")
    print()

    return ds


if __name__ == "__main__":
    fire.Fire(prepare_dataset)
