import os
from datasets import Dataset, load_dataset, concatenate_datasets
from transformers import AutoTokenizer

from datetime import datetime, timezone

started_time = datetime.now(timezone.utc)
started_time_str = started_time.strftime('%Y-%m-%d-%H-%M-%S')
file_dir = os.path.dirname(os.path.abspath(__file__))
datasets_path = os.path.join(file_dir, 'datasets')
os.makedirs(datasets_path, exist_ok=True)

# Config
tokenizer_name = 'zetavg/test-pythia-zh-tw-tokenizer-50000-20230507'
cutoff_len = 2048
single_dataset_rows_limit = 300000
# single_dataset_rows_limit = 1000
preview_length = 64
# preview_length = 1024
dataset_general_name = 'wiki-trans-t'

limit_name = ''
if single_dataset_rows_limit:
    limit_name = f'-lm{single_dataset_rows_limit}'
dataset_name = f"tds-{tokenizer_name.replace('/', '-')}-{dataset_general_name}-c{cutoff_len}{limit_name}"

print(f"Loading tokenizer '{tokenizer_name}'...")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

# if no pad token, set it to eos
if tokenizer.pad_token is None:
    print(
        f"Tokenizer has no pad_token set, setting it to 1.")
    tokenizer.pad_token_id = 1
print(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}.")


def get_tokenize_data_fn(dataset_column):
    def tokenize_data(data_point):
        batch_encoding = tokenizer(
            # See: https://huggingface.co/docs/transformers/main/en/main_classes/tokenizer#tokenizer
            data_point[dataset_column],
            max_length=cutoff_len,
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
                batch_encoding['preview'].append(source_text[:preview_length])
        else:
            batch_encoding["labels"] = batch_encoding["input_ids"].copy()
            batch_encoding["preview"] = source_text[:preview_length]
        return batch_encoding
    return tokenize_data


print('Loading wikipedia dataset...')
wiki_ds: Dataset = load_dataset(
    'zetavg/zh-tw-wikipedia')['train']    # type: ignore
print('Processing wikipedia dataset...')
wiki_ds = wiki_ds.shuffle()
if single_dataset_rows_limit:
    wiki_ds = wiki_ds.select(range(single_dataset_rows_limit))
wiki_train_ds = wiki_ds.map(
    get_tokenize_data_fn('markdown'),
    remove_columns=list(wiki_ds.features.keys()),
    desc="Tokenizing Wikipedia dataset",
    batched=True,
    batch_size=512,
)
wiki_train_ds = wiki_train_ds.filter(
    lambda x: len(x["input_ids"]) > 0,
    # batched=True
    )


print('Loading translations dataset...')
trans_ds: Dataset = load_dataset(
    'zetavg/coct-en-zh-tw-translations-twp-300k')['train']    # type: ignore
print('Processing translations dataset...')
trans_ds = trans_ds.shuffle()
if single_dataset_rows_limit:
    trans_ds = trans_ds.select(range(single_dataset_rows_limit))
en_first = True
def get_translations_text(data_point):
    global en_first
    if en_first:
        text = f"English: {data_point['en']}\nChinese: {data_point['ch']}"
    else:
        text = f"Chinese: {data_point['ch']}\nEnglish: {data_point['en']}"

    en_first = not en_first
    return { 'text': text.strip() }
trans_train_ds = trans_ds.map(get_translations_text).map(
    get_tokenize_data_fn('text'),
    remove_columns=list(trans_ds.features.keys()) + ['text'],
    desc="Tokenizing translations dataset",
    batched=True,
    batch_size=512,
)
trans_train_ds = trans_train_ds.filter(
    lambda x: len(x["input_ids"]) > 0,
    # batched=True
    )

print('Generating merged dataset...')
train_ds = concatenate_datasets([wiki_train_ds, trans_train_ds]).shuffle()

print('Saving dataset...')
train_ds.save_to_disk(os.path.join(datasets_path, dataset_name))

print('Pushing dataset to HF Hub...')
train_ds.push_to_hub(dataset_name, private=True)
