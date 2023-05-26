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
    PeftModel,
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
from utils.get_torch_dtype_from_str import get_torch_dtype_from_str
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


def main(
    train_name: str,
    cfg: Union[str, None] = None,
    config_file_path: Union[str, None] = None,
    data_dir_path: Union[str, None] = None,
):
    global early_abort

    paths = Paths(data_dir_path)
    if cfg and not config_file_path:
        config_file_path = paths.get_config_path(cfg)
    config = Config(config_file_path)

    training_config = config.get_training_config(train_name)

    (
        model_name, torch_dtype,
        base_model_name, tokenizer_name, base_on_model_name,
        dataset_name,
        peft_type
    ) = map(
        get_training_config_values(config, training_config).get,
        (
            'model_name', 'torch_dtype',
            'base_model_name', 'tokenizer_name', 'base_on_model_name',
            'dataset_name',
            'peft_type'
        ))

    if peft_type != 'lora':
        raise ValueError('This only works with peft_type: lora.')

    base_on_model_name_or_path: str = base_on_model_name  # type: ignore
    possible_model_path = paths.get_model_path(base_on_model_name_or_path)
    if os.path.isdir(possible_model_path):
        base_on_model_name_or_path = possible_model_path
    elif '/' not in base_on_model_name_or_path:
        base_on_model_name_or_path = \
            f"{config.hf_user_or_org_name}/{base_on_model_name_or_path}"

    lora_model_name_or_path: str = model_name
    possible_lora_model_path = paths.get_model_path(model_name)
    if os.path.isdir(possible_lora_model_path):
        lora_model_name_or_path = possible_lora_model_path
    elif '/' not in lora_model_name_or_path:
        lora_model_name_or_path = \
            f"{config.hf_user_or_org_name}/{lora_model_name_or_path}"

    print(f"Merge LoRA model")
    print()
    print(
        colored("Base model:", 'cyan'),
        base_on_model_name_or_path,
        f"({torch_dtype})" if torch_dtype else '')
    print(colored("LoRA model:", 'cyan'), lora_model_name_or_path)
    print(colored("Tokenizer:", 'cyan'), tokenizer_name)
    print()

    tokenizer = load_tokenizer(config, paths)

    print(f"Loading base model '{base_on_model_name_or_path}'...")
    model = AutoModelForCausalLM.from_pretrained(
        base_on_model_name_or_path,
        torch_dtype=get_torch_dtype_from_str(torch_dtype),
        device_map='auto'
        )

    print(f"Loading LoRA model '{lora_model_name_or_path}'...")
    model = PeftModel.from_pretrained(
        model,
        lora_model_name_or_path,
        device_map='auto'
    )

    print("Merging LoRA model...")
    model = model.merge_and_unload()

    merged_model_name = model_name + '-merged'
    merged_model_output_path = paths.get_model_path(merged_model_name)
    print(f"Saving merged model as '{merged_model_output_path}...")

    model.save_pretrained('tmp/merged_model')
    print('...')
    model.save_pretrained(merged_model_output_path)
    tokenizer.save_pretrained(merged_model_output_path)
    print(colored(
        f"Model saved to {merged_model_output_path}.",
        attrs=['bold']
    ))
    print()

    if config.push_outputs_to_hf:
        hf_model_name = f"{config.hf_user_or_org_name}/{merged_model_name}"

        print("Pushing to HF Hub...")
        results = model.push_to_hub(
            hf_model_name,
            private=True
        )
        print(results)
        results = tokenizer.push_to_hub(
            hf_model_name,
            private=True
        )
        print(results)
        print("Updating model card...")
        model_card_frontmatter = {
            # 'datasets': [hf_dataset_name],
        }
        model_card_content = dedent(f"""
        # {merged_model_name}

        This model is a part of the `{config.project_name}` project.
        """).strip()
        model_card_content += '\n\n'
        model_card_content += dedent(f"""
        * Base model: `{base_on_model_name}`
        * LoRA model: `{model_name}`
        * Tokenizer: `{tokenizer_name}`
        """).strip()
        update_hf_readme(hf_model_name, model_card_content,
                         model_card_frontmatter)
        print(colored(
            f"Model uploaded to https://huggingface.co/{hf_model_name}.",
            attrs=['bold']
        ))
        print()

    print(colored(
        "Done.",
        'green',
        attrs=['bold']
    ))


if __name__ == "__main__":
    fire.Fire(main)
