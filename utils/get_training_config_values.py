from typing import TypedDict, Dict

from termcolor import colored


def get_training_config_values(
        config, training_config, print_values=False) -> Dict[str, str]:
    model_name = training_config.model_name
    base_model_name = config.base_model_name
    torch_dtype = config.torch_dtype
    tokenizer_name = config.tokenizer_name
    train_name = training_config.config_name
    dataset_name = training_config.dataset_name

    base_on_model_name = training_config.base_on_model_name

    peft_type = training_config.use_peft

    # training_args = TrainingArguments(training_config.training_args)
    # peft_config = None
    # if peft_type == 'lora':
    #     peft_config = LoraConfig(**training_config._config['lora_config'])

    if print_values:
        print()
        print(colored("Run name:", 'cyan'), training_config.run_name)
        print(colored("Output model name:", 'cyan'), model_name)
        print()
        print(
            colored("Base model:", 'cyan'),
            base_model_name,
            f"({torch_dtype})" if torch_dtype else '')
        print(colored("Tokenizer:", 'cyan'), tokenizer_name)
        print()

        if base_on_model_name != base_model_name:
            print(colored("Base on model:", 'cyan'), base_on_model_name)
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
        'base_on_model_name': base_on_model_name,
        'train_name': train_name,
        'dataset_name': dataset_name,
        'peft_type': peft_type,
        'torch_dtype': torch_dtype,
    }
