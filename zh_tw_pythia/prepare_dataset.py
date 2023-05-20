from typing import Any, Union

import os
import re
import fire
import json
from transformers import AutoTokenizer
from tokenizers import Tokenizer as TokenizerFast
from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets
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
    test_datasets = []

    for build_type in dataset_config.build_with:
        settings = dataset_config.get_settings_for(build_type)
        source_ds_name = settings.get('source_dataset')
        rows_limit = settings.get('rows_limit')
        test_size = settings.get('test_size')
        test_split_seed = settings.get('test_split_seed')
        test_rows_limit = settings.get('test_rows_limit')
        print(f"Loading source dataset '{source_ds_name}'...")
        source_ds: Dataset = \
            load_dataset(source_ds_name)['train']  # type: ignore
        print(f"Source dataset contains {len(source_ds)} rows.")
        test_ds = None
        if test_size:
            print('Splitting dataset into train and test sets...')
            ds_dict = source_ds.train_test_split(
                test_size=test_size,
                shuffle=False,
                seed=test_split_seed,
            )
            source_ds = ds_dict['train']
            test_ds = ds_dict['test']
            print(
                f"Train set has {len(source_ds)} rows, test set has {len(test_ds)} rows.")
        if build_type == 'translations':
            generate_dataset_fn = generate_translations_dataset

        elif build_type == 'wikipedia':
            generate_dataset_fn = generate_wikipedia_dataset

        elif build_type == 'sharegpt':
            generate_dataset_fn = generate_sharegpt_dataset

        elif build_type == 'alpaca':
            generate_dataset_fn = generate_alpaca_dataset

        else:
            raise Exception(
                f"Unknown dataset build method '{build_type}'. Check {dataset_config.config_file_path}")

        ds = generate_dataset_fn(
            tokenizer, dataset_config, settings,
            source_ds, rows_limit)
        datasets.append(ds)
        if test_ds:
            print("Preparing test set...")
            t_ds = generate_dataset_fn(
                tokenizer, dataset_config, settings,
                test_ds, test_rows_limit, type_='test')
            test_datasets.append(t_ds)
        print()

    print("Preparing final dataset...")
    dataset = concatenate_datasets(datasets)
    print(f"Concatenated dataset contains {len(dataset)} rows.")

    dataset = dataset.filter(lambda x: x['input_ids'])

    sort_by = dataset_config.sort_by
    if sort_by:
        column, order = sort_by
        print(f"Sort by '{column}' {order}...")
        dataset = dataset.sort(column, reverse=order == 'desc')
    else:
        dataset = dataset.shuffle()
    if dataset_config._config.get('only_first_n_rows'):
        dataset = dataset.select(range(dataset_config._config['only_first_n_rows']))
    print(colored(
        f"Final dataset contains {len(dataset)} rows.",
        attrs=['bold']
    ))
    print()

    test_dataset = None
    if test_datasets:
        print("Preparing final test dataset...")
        test_dataset = concatenate_datasets(test_datasets)
        print(f"Concatenated test dataset contains {len(test_dataset)} rows.")

        test_dataset = test_dataset.filter(lambda x: x['input_ids'])
        print(colored(
            f"Final test dataset contains {len(test_dataset)} rows.",
            attrs=['bold']
        ))
        print()
        dataset = DatasetDict({
            'train': dataset,
            'test': test_dataset,
        })

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
        rows_info = f"`train` `{len(dataset['train'])}`, `test` `{len(dataset['test'])}`" if isinstance(
            dataset, DatasetDict) else f"`{len(dataset)}`"
        dataset_card_content = dedent(f"""
        # {dataset_config.dataset_name}

        This dataset is a part of the `{config.project_name}` project.

        * Tokenizer: `{config.tokenizer_name}`
        * Built with: {', '.join(f"`{s}`" for s in dataset_config.build_with)}
        * Rows: {rows_info}
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
            batch_encoding['length'] = []
            for i, source_text in enumerate(data_point[dataset_column]):
                batch_encoding['labels'].append(
                    batch_encoding['input_ids'][i].copy())
                preview_text = source_text
                if len(preview_text) > preview_length:
                    preview_text = preview_text[:preview_length] + ' [...]'
                batch_encoding['preview'].append(preview_text)
                batch_encoding['length'].append(len(batch_encoding['input_ids'][i]))
        else:
            # not batched
            batch_encoding["labels"] = batch_encoding["input_ids"].copy()
            preview_text = source_text
            if len(preview_text) > preview_length:
                preview_text = preview_text[:preview_length] + ' [...]'
            batch_encoding["preview"] = preview_text
            batch_encoding['length'] = len(batch_encoding["input_ids"])
        return batch_encoding
    return tokenize_data


def generate_translations_dataset(tokenizer, dataset_config, settings, source_ds, rows_limit, type_='train'):
    lang_1_key = settings.get('lang_1_key')
    assert lang_1_key, f"{dataset_config.get_config_level_str(['translations_settings', 'lang_1_key'])} is missing in config {dataset_config.config_file_path}."
    lang_2_key = settings.get('lang_2_key')
    assert lang_2_key, f"{dataset_config.get_config_level_str(['translations_settings', 'lang_2_key'])} is missing in config {dataset_config.config_file_path}."
    templates = settings.get('templates')
    assert templates, f"{dataset_config.get_config_level_str(['translations_settings', 'templates'])} is missing in config {dataset_config.config_file_path}."
    use_template = settings.get('use_template')

    print('Processing translations dataset...')

    if type_ != 'test':
        source_ds = source_ds.shuffle()

    if rows_limit:
        print(f"Limiting to {rows_limit} rows.")
        source_ds = source_ds.select(range(rows_limit))

    t_i = 0
    def get_translations_text(batch):
        nonlocal t_i
        output = {'text': []}

        for lang_1_text, lang_2_text in zip(
                batch[lang_1_key], batch[lang_2_key]):

            ts = templates
            if use_template == 'random':
                # Not actually random, we need a same output for different runs.
                if t_i >= len(templates):
                    t_i = 0
                ts = [templates[t_i]]
                t_i += 1

            for template in ts:
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

    print(colored(
        f"Translations {type_} dataset ok. Has {len(ds)} rows.",
        attrs=['bold']
    ))

    return ds


def generate_wikipedia_dataset(tokenizer, dataset_config, settings, source_ds, rows_limit, type_='train'):
    exclude = settings.get('exclude')

    print('Processing wikipedia dataset...')

    if type_ != 'test':
        source_ds = source_ds.shuffle()

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

    print(
        'Sample titles:',
        [d.get('title') or d.get('original_title')
         for d in source_ds.select(range(min(100, len(source_ds))))])

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

    print(colored(
        f"Wikipedia {type_} dataset ok. Has {len(ds)} rows.",
        attrs=['bold']
    ))

    return ds


def generate_sharegpt_dataset(tokenizer, dataset_config, settings, source_ds, rows_limit, type_='train'):
    languages = settings.get('languages')
    train_on_inputs = settings.get('train_on_inputs')

    print('Processing ShareGPT dataset...')

    unknown_from_values = []

    def has_gpt(row):
        for c in row['conversations']:
            if c['from'] == 'gpt' or c['from'] == 'chatgpt':
                return True
            elif c['from'] != 'human' and c['from'] != 'user':
                if c['from'] not in unknown_from_values:
                    unknown_from_values.append(c['from'])
        return False

    source_ds = source_ds.filter(has_gpt)
    if unknown_from_values:
        print(f"Unknown 'from' values: {unknown_from_values}")

    if type_ != 'test':
        source_ds = source_ds.shuffle()

    if not rows_limit or rows_limit > len(source_ds):
        rows_limit = len(source_ds)
    lang_limits = {}
    if languages:
        lang_limits = {
            list(lang.keys())[0]: list(lang.values())[0]  # type: ignore
            for lang in languages if isinstance(lang, dict)}
        lang_limits = {k: v if v >= 1 else int(
            rows_limit * v) for k, v in lang_limits.items()}
    lang_counts = {}
    languages = [
        la if isinstance(la, str) else list(la.keys())[0]
        for la in languages]

    def data_generator():
        progress_bar = tqdm(total=rows_limit)
        rows_yield = 0
        i = 0
        while rows_yield < rows_limit and i < len(source_ds):
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
        if f == 'human' or f == 'user':
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

        elif f == 'gpt' or f == 'chatgpt':
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
        messages_count = 0
        for c in data_point['conversations']:
            message = c['opencc_converted_markdown'] or c['markdown']
            message = message.strip()
            message = message.strip('\u200b')
            if not message:
                continue
            result = tokenize_message(message, c['from'])
            if not result:
                continue

            if (len(output['input_ids']) + len(result['input_ids']) + len(ending_result['input_ids'])) > dataset_config.max_tokens_length:
                break

            if c['from'] == 'human' or c['from'] == 'user':
                last_is_input = True
            else:
                last_is_input = False

            output['input_ids'] += result['input_ids']
            output['attention_mask'] += result['attention_mask']
            output['labels'] += result['labels']
            messages_count += 1

        if not last_is_input:
            output['input_ids'] += ending_result['input_ids']
            output['attention_mask'] += ending_result['attention_mask']
            output['labels'] += ending_result['input_ids']

        if len(output['input_ids']) > dataset_config.max_tokens_length:
            output['input_ids'] = output['input_ids'][:dataset_config.max_tokens_length]
            output['attention_mask'] = output['attention_mask'][:dataset_config.max_tokens_length]
            output['labels'] = output['labels'][:dataset_config.max_tokens_length]

        preview_length = dataset_config.preview_length
        preview_text = f"[{data_point['id']}] " + \
            tokenizer.decode(output['input_ids'])
        if len(preview_text) > preview_length:
            preview_text = preview_text[:preview_length] + ' [...]'
        output["preview"] = preview_text
        output['length'] = len(output['input_ids'])
        output['messages_count'] = messages_count

        return output

    ds = ds.map(
        tokenize_data,
        remove_columns=list(source_ds.features.keys())
    )

    ds = ds.filter(lambda x: x['length'] > 8 and x['messages_count'] >= 2)

    print(colored(
        f"ShareGPT {type_} dataset ok. Has {len(ds)} rows.",
        attrs=['bold']
    ))

    return ds


def generate_alpaca_dataset(tokenizer, dataset_config, settings, source_ds, rows_limit, type_='train'):
    template = settings.get('template')
    train_on_inputs = settings.get('train_on_inputs')

    print('Processing alpaca dataset...')

    if type_ != 'test':
        source_ds = source_ds.shuffle()

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
        result['length'] = len(result["input_ids"])

        return result

    ds = ds.map(
        tokenize_data,
        remove_columns=['prompt', 'completion'],
    )

    print(colored(
        f"Alpaca {type_} dataset ok. Has {len(ds)} rows.",
        attrs=['bold']
    ))

    return ds


if __name__ == "__main__":
    fire.Fire(prepare_dataset)
