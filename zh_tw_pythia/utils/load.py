import os
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from datasets import (
    DatasetDict, load_dataset as load_hf_dataset, load_from_disk
)


def load_tokenizer(config, paths) -> PreTrainedTokenizerFast:
    tokenizer = None

    local_tokenizer_path = paths.get_tokenizer_path(config.tokenizer_name)
    hf_hub_tokenizer_name = config.tokenizer_name
    if '/' not in hf_hub_tokenizer_name:
        hf_hub_tokenizer_name = \
            f"{config.hf_user_or_org_name}/{config.tokenizer_name}"
    if os.path.isdir(local_tokenizer_path):
        print(f"Loading tokenizer from local path '{local_tokenizer_path}'...")
        tokenizer = AutoTokenizer.from_pretrained(local_tokenizer_path)
    else:
        print(f"Loading tokenizer from HF '{hf_hub_tokenizer_name}'...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(hf_hub_tokenizer_name)
        except Exception as e:
            print(e)

    if not tokenizer:
        raise Exception(
            f"Can't load tokenizer from either HuggingFace Hub ({hf_hub_tokenizer_name}) or local path ({local_tokenizer_path}).")

    # if no pad token, set it to eos
    if tokenizer.pad_token is None:
        print(
            f"Tokenizer has no pad_token set, setting it to 1.")
        tokenizer.pad_token_id = 1

    print(
        f"Tokenizer '{tokenizer.__class__.__name__}' loaded. Vocab size: {tokenizer.vocab_size}.")
    print()

    return tokenizer


def load_dataset(config, paths, dataset_name) -> DatasetDict:
    dataset = None

    local_dataset_path = paths.get_dataset_path(dataset_name)
    hf_hub_dataset_name = dataset_name
    if '/' not in hf_hub_dataset_name:
        hf_hub_dataset_name = \
            f"{config.hf_user_or_org_name}/{dataset_name}"
    if os.path.isdir(local_dataset_path):
        print(f"Loading dataset from local path '{local_dataset_path}'...")
        dataset = load_from_disk(local_dataset_path)
        if not isinstance(dataset, DatasetDict):
            dataset = {'train': dataset}
    else:
        print(f"Loading dataset from HF '{hf_hub_dataset_name}'...")
        try:
            dataset = load_hf_dataset(hf_hub_dataset_name)
        except Exception as e:
            print(e)

    if not dataset:
        raise Exception(
            f"Can't load dataset from either HuggingFace Hub ({hf_hub_dataset_name}) or local path ({local_dataset_path}).")

    print(
        f"Dataset '{dataset_name}' loaded.")
    print()

    return dataset  # type: ignore
