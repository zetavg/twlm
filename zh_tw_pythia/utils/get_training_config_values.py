from typing import TypedDict, Dict

from termcolor import colored
from peft import (
    LoraConfig,
)

# ReturnValue = TypedDict(
#     'ReturnValue',
#     {
#         'model_name': str,
#         'base_model_name': str,
#         'tokenizer_name': str,
#         'train_name': str,
#         'dataset_name': str,
#     }
# )


def get_training_config_values(
        config, training_config, print_values=False) -> Dict[str, str]:
    model_name = training_config.model_name
    base_model_name = config.base_model_name
    tokenizer_name = config.tokenizer_name
    train_name = training_config.config_name
    dataset_name = training_config.dataset_name

    base_on = training_config.base_on
    if base_on:
        if base_on.get('model'):
            base_model_name = base_on['model']
        elif base_on.get('output_of'):
            based_t_cfg = config.get_training_config(base_on['output_of'])
            base_model_name = based_t_cfg.model_name
        else:
            raise ValueError(f"'{base_on}' is not a valid 'base_on' value.")

    peft_type = training_config.use_peft

    if print_values:
        print()
        print(colored("Output model name:", 'cyan'), model_name)
        print()
        print(colored("Base model:", 'cyan'), base_model_name)
        print(colored("Tokenizer:", 'cyan'), tokenizer_name)
        print()

        if peft_type:
            print(colored("PEFT method:", 'cyan'), peft_type)
            print()

        print(colored("Train:", 'cyan'), train_name)
        print(colored("Dataset:", 'cyan'), dataset_name)
        print()

    return {
        'model_name': model_name,
        'base_model_name': base_model_name,
        'tokenizer_name': tokenizer_name,
        'train_name': train_name,
        'dataset_name': dataset_name,
        'peft_type': peft_type,
    }
