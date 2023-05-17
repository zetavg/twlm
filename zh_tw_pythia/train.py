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

    model_name, base_model_name, tokenizer_name, base_on_model_name, dataset_name, peft_type = map(
        get_training_config_values(config, training_config).get,
        ('model_name', 'base_model_name', 'tokenizer_name', 'base_on_model_name', 'dataset_name', 'peft_type'))

    model_output_path = paths.get_model_path(model_name)

    base_on_model_name_or_path: str = base_on_model_name  # type: ignore
    possible_model_path = paths.get_model_path(base_on_model_name_or_path)
    if os.path.isdir(possible_model_path):
        base_on_model_name_or_path = possible_model_path
    elif '/' not in base_on_model_name_or_path:
        base_on_model_name_or_path = \
            f"{config.hf_user_or_org_name}/{base_on_model_name_or_path}"

    print(f"Starting train '{training_config.run_name}'...")
    print()
    print(colored("Base on model:", 'cyan'), base_on_model_name_or_path)
    print(colored("Tokenizer:", 'cyan'), tokenizer_name)
    print()
    if peft_type:
        print(colored("PEFT method:", 'cyan'), model_output_path)
        print()
    print(colored("Train:", 'cyan'), training_config.config_name)
    print(colored("Dataset:", 'cyan'), dataset_name)
    print()
    print(colored("Output path:", 'cyan'), model_output_path)
    print()

    run_tags = [
        f"group:{config.group_name}"[:64],
        f"train:{training_config.config_name}"[:64],
        f"bm:{base_model_name}"[:64],
        f"bom:{base_on_model_name}"[:64],
        f"tokenizer:{tokenizer_name}"[:64],
        f"ds:{dataset_name}"[:64],
    ]

    use_wandb = config.report_to_wandb

    if use_wandb:
        wandb.init(
            project=config.wandb_project,
            group=config.wandb_group,
            name=training_config.run_name,
            resume="allow",
            # Unique ID for resuming
            id=f"{config.wandb_project}--{training_config.run_name}",
            tags=run_tags,
            save_code=True,
            magic=True,
        )
        wandb.config.update({
            'training_config': training_config._config,
            'config': config._config,
        }, allow_val_change=True)
        wandb.config.update({
            'base_model': base_model_name,
            'base_on_model': base_on_model_name,
            'tokenizer': tokenizer_name,
            'train': training_config.config_name,
            'dataset': dataset_name,
            'output_model_name': model_name,
        })
        print()

    resume_from_checkpoint = find_checkpoint_to_resume(model_output_path)
    if resume_from_checkpoint:
        possible_training_args_path = \
            os.path.join(resume_from_checkpoint, 'training_args.bin')
        if os.path.isfile(possible_training_args_path):
            # Allows the training args to be changed.
            os.rename(
                possible_training_args_path,
                os.path.join(resume_from_checkpoint, 'old_training_args.bin')
            )

    tokenizer = load_tokenizer(config, paths)

    dataset = load_dataset(config, paths, dataset_name)
    train_dataset = dataset['train']
    test_dataset = dataset.get('test', [])
    print(f"Train dataset contains {len(train_dataset)} rows.")
    if test_dataset:
        print(f"Test dataset contains {len(test_dataset)} rows.")
    print()

    print(f"Loading base model '{base_on_model_name_or_path}'...")
    model = AutoModelForCausalLM.from_pretrained(
        base_on_model_name_or_path, device_map='auto')
    print(
        f"Base model loaded, input_embeddings: {model.get_input_embeddings()}, output_embeddings: {model.get_output_embeddings()}.")
    print()

    if (
        model.get_input_embeddings().num_embeddings != tokenizer.vocab_size or
        model.get_output_embeddings().out_features != tokenizer.vocab_size
    ):
        print(f"Resizing model to match tokenizer vocab size...")

        original_all_params_count = 0
        for name, param in model.named_parameters():
            original_all_params_count += param.numel()

        original_input_embeddings_parameters_count = sum([
            p[1].numel()
            for p in model.get_input_embeddings().named_parameters()])
        original_output_embeddings_parameters_count = sum([
            p[1].numel()
            for p in model.get_output_embeddings().named_parameters()])

        model.resize_token_embeddings(tokenizer.vocab_size)

        new_all_params_count = 0
        for name, param in model.named_parameters():
            new_all_params_count += param.numel()

        new_input_embeddings_parameters_count = sum([
            p[1].numel()
            for p in model.get_input_embeddings().named_parameters()])
        new_output_embeddings_parameters_count = sum([
            p[1].numel()
            for p in model.get_output_embeddings().named_parameters()])

        print(
            f"New input_embeddings: {model.get_input_embeddings()}, output_embeddings: {model.get_output_embeddings()}.")
        print(
            f"Original input embeddings / output embeddings / all params count: {original_input_embeddings_parameters_count} / {original_output_embeddings_parameters_count} / {original_all_params_count}")
        print(
            f"New input embeddings / output embeddings / all params count: {new_input_embeddings_parameters_count} / {new_output_embeddings_parameters_count} / {new_all_params_count}")
        print()

    train_params = training_config.only_train_parameters_matching
    if train_params:
        print(f"Will only train params matching: {', '.join(train_params)}.")
        trainable_params_list = []
        frozen_params_list = []
        for name, param in model.named_parameters():
            if not any(re.search(pattern, name) for pattern in train_params):
                frozen_params_list.append(name)
                param.requires_grad = False
            else:
                trainable_params_list.append(name)
        print()
        print("trainable_params:", trainable_params_list)
        print()
        # print("frozen_params:", frozen_params_list)
        # print()
        if use_wandb:
            wandb.config.update({
                'only_train_parameters_matching': train_params,
                'trainable_params': trainable_params_list,
                'frozen_params': frozen_params_list,
            })

    if peft_type:
        print("Creating PEFT model...")
        print()
        if peft_type == 'lora':
            peft_config = LoraConfig(**training_config._config['lora_config'])
            model = get_peft_model(model, peft_config)
        else:
            raise ValueError(f"Unknown PEFT method: {peft_type}.")

    trainable_params_count = 0
    all_params_count = 0
    for _, param in model.named_parameters():
        all_params_count += param.numel()
        if param.requires_grad:
            trainable_params_count += param.numel()
    trainable_params_rate = trainable_params_count / all_params_count
    print(
        f"trainable params: {trainable_params_count} || all params: {all_params_count} || trainable%: {100 * trainable_params_rate}"
    )
    print()
    if use_wandb:
        wandb.config.update({
            'all_params_count': all_params_count,
            'trainable_params_count': trainable_params_count,
            'trainable_params_rate': trainable_params_rate,
        })

    print("Base model ready.")
    print()

    data_collator = DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    )

    # See: https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments
    training_args = TrainingArguments(**{
        'output_dir': model_output_path,
        'overwrite_output_dir': True,
        'report_to': ['wandb'] if use_wandb else None,
        'evaluation_strategy': 'steps' if len(test_dataset) > 0 else 'no',
        **training_config.training_arguments,
        'eval_steps': (
            training_config.training_arguments.get('eval_steps') or
            training_config.training_arguments.get('save_steps') or
            10
        ) if len(test_dataset) > 0 else None,
    })

    if use_wandb:
        training_args_dict = training_args.to_dict()
        wandb.config.update({
            'training_arguments': {
                k: v for k, v in training_args_dict.items()
                if k not in training_config.training_argument_keys_allow_updating
            },
        })
        wandb.config.update({
            'training_arguments_other': {
                k: v for k, v in training_args_dict.items()
                if k in training_config.training_argument_keys_allow_updating
            },
        }, allow_val_change=True)

    # See: https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.Trainer
    trainer = TrainerWithOutputLogging(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,  # type: ignore
        eval_dataset=test_dataset,  # type: ignore
        data_collator=data_collator,
        args=training_args,
        callbacks=[TrainerControlCallback],  # type: ignore
    )
    trainer._output_logging_tokenizer = tokenizer  # type: ignore
    trainer.log_output_every_n_steps = \
        training_config._config.get('log_output_every_n_steps') \
        or (training_args.logging_steps * 20)  # type: ignore

    if resume_from_checkpoint:
        if isinstance(resume_from_checkpoint, str):
            print(colored(
                f"Resuming from checkpoint '{resume_from_checkpoint}'...",
                'green',
                attrs=['bold']
            ))
            print()
        else:
            print(colored(
                "Resuming from latest checkpoint...",
                'green',
                attrs=['bold']
            ))
            print()
    else:
        print(colored(
            "Train starting...",
            attrs=['bold']
        ))
        print()

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    print()
    print(f"Saving model to {model_output_path}...")
    model.save_pretrained(model_output_path)
    tokenizer.save_pretrained(model_output_path)
    if early_abort:
        with open(os.path.join(model_output_path, 'early_abort.json'), 'w') as f:
            json.dump(early_abort, f, indent=2, ensure_ascii=False)
    print(colored(
        f"Model saved to {model_output_path}.",
        attrs=['bold']
    ))
    print()

    if config.push_outputs_to_hf:
        hf_model_name = f"{config.hf_user_or_org_name}/{model_name}"
        hf_dataset_name = f"{config.hf_user_or_org_name}/{dataset_name}"

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
            'datasets': [hf_dataset_name],
        }
        model_card_content = dedent(f"""
        # {model_name}

        This model is a part of the `{config.project_name}` project.
        """).strip()
        if early_abort:
            model_card_content += '\n\n'
            model_card_content += dedent(f"""
            **Training has been early aborted at `epoch` `{early_abort.get('epoch')}`, `global_step` `{early_abort.get('global_step')}`.**
            """).strip()
        model_card_content += '\n\n'
        model_card_content += dedent(f"""
        * Base model: `{base_on_model_name_or_path}`
        * Tokenizer: `{tokenizer_name}`
        * Vocab size: `{tokenizer.vocab_size}`
        * Train: `{training_config.config_name}`
        * Dataset used: `{dataset_name}`
        * Full config:
          ```json
          {config.to_json()}
          ```
        """).strip()
        update_hf_readme(hf_model_name, model_card_content,
                         model_card_frontmatter)
        print(colored(
            f"Model uploaded to https://huggingface.co/{hf_model_name}.",
            attrs=['bold']
        ))
        print()

    if use_wandb:
        wandb.finish()
        print()
        print(colored(
            f"Model saved at: {model_output_path}.",
            attrs=['bold']
        ))
        if config.push_outputs_to_hf:
            print(colored(
                f"Model on HF Hub: https://huggingface.co/{hf_model_name}.",
                attrs=['bold']
            ))

    print()
    print(colored(
        "Done.",
        'green',
        attrs=['bold']
    ))


class TrainerWithOutputLogging(Trainer):
    def training_step(self, model, inputs):
        tensor = super().training_step(model, inputs)

        if hasattr(self, "_current_step_for_output_logging"):
            self._current_step_for_output_logging += 1
        else:
            self._current_step_for_output_logging = 0

        return tensor

    def compute_loss(self, model, inputs, return_outputs=False):

        should_compute_loss_return_outputs = return_outputs
        should_log_output = False

        if hasattr(self, "_current_step_for_output_logging"):
            if self._current_step_for_output_logging % self.log_output_every_n_steps == 0:  # type: ignore
                # force the original `training_step` to return outputs
                # so we can inspect it
                should_compute_loss_return_outputs = True
                should_log_output = True

        compute_loss_result = super().compute_loss(
            model, inputs,
            return_outputs=should_compute_loss_return_outputs
        )

        if should_log_output:
            loss, outputs = compute_loss_result
            try:
                tokenizer = self._output_logging_tokenizer  # type: ignore
                # Preview what the model have generated
                logits = outputs.logits  # type: ignore
                # Get the token IDs with the highest probabilities
                token_ids = logits.argmax(dim=-1).squeeze().tolist()
                if isinstance(token_ids[0], list):
                    # is in a batch, get thee first one
                    token_ids = token_ids[0]

                limit = 1024
                labels = inputs['labels'][0]
                labels_to_decode = labels[:limit]
                labels_to_decode = labels_to_decode.tolist()
                token_ids_to_decode = token_ids[:len(labels_to_decode)]

                while labels_to_decode and labels_to_decode[-1] == -100:
                    labels_to_decode.pop()
                    token_ids_to_decode.pop()

                if labels_to_decode[0] == -100:
                    while labels_to_decode and labels_to_decode[1] == -100:
                        labels_to_decode.pop(0)
                        token_ids_to_decode.pop(0)

                label_tokens: List[str] = [
                    tokenizer.decode([i]) if i >= 0 else ''
                    for i in labels_to_decode]
                output_tokens: List[str] = [
                    tokenizer.decode([i])
                    for i in token_ids_to_decode]
                output_tokens = output_tokens

                # Will be in WandB logs anyway.
                # self.log({  # type: ignore
                #     'output_tokens': output_tokens,
                #     'label_tokens': label_tokens,
                #     'output_ids': token_ids,
                #     'labels': labels.tolist()
                # })
                print(colored(
                    '----------------',
                    'dark_grey',
                ))
                input_preview = inputs['input_ids'][0].tolist()
                input_preview_truncated = False
                while input_preview and input_preview[-1] <= 1:
                    input_preview.pop()
                if len(input_preview) > 80:
                    input_preview = input_preview[:80]
                    input_preview_truncated = True
                text = tokenizer.decode(input_preview).replace('\n', '\\n')
                text += ' [...]' if input_preview_truncated else ''
                print('"' + text + '"')
                print(colored(
                    '----------------',
                    'dark_grey',
                ))
                print(comparing_lists(
                    [
                        label_tokens,
                        [''] + output_tokens,
                        labels_to_decode,
                        [''] + token_ids_to_decode
                    ],
                    labels=['Labels', 'Outputs', '', ''],
                    colors=[None, None, 'dark_grey', 'dark_grey'],
                    add_blank_line=False,
                ))
                print(colored(
                    '----------------',
                    'dark_grey',
                ))
            except Exception as e:
                print("inputs:", inputs)
                print("compute_loss_result:", compute_loss_result)
                print("Failed to log output:", str(e))
                traceback.print_tb(e.__traceback__)

        if should_compute_loss_return_outputs:
            loss, outputs = compute_loss_result
            return (loss, outputs) if return_outputs else loss
        else:
            return compute_loss_result


def find_checkpoint_to_resume(output_dir):
    if not os.path.isdir(output_dir):
        return False

    checkpoints = [
        os.path.join(output_dir, d)
        for d in os.listdir(output_dir) if d.startswith("checkpoint")
    ]
    if len(checkpoints) <= 0:
        return False

    print(
        f"Found {len(checkpoints)} checkpoints in {output_dir}.")

    # Filter checkpoints containing 'trainer_state.json', to prevent resuming
    # from a checkpoint that is not fully saved.
    filtered_checkpoints = [
        ckpt for ckpt in checkpoints
        if os.path.isfile(os.path.join(ckpt, 'trainer_state.json'))]
    if len(filtered_checkpoints) <= 0:
        print(colored(
            "Non of the checkpoints are valid, will not resume from checkpoint.",
            'yellow',
            attrs=['bold']
        ))
        return False

    # Find the latest checkpoint.
    checkpoints_number_matches = [
        re.search(r'checkpoint-(\d+)$', ckpt)
        for ckpt in filtered_checkpoints]
    checkpoints_numbers = [int(m.group(1))
                           for m in checkpoints_number_matches if m]
    if len(checkpoints_numbers) <= 0:
        print(colored(
            "Non of the checkpoints are valid, will not resume from checkpoint.",
            'yellow',
            attrs=['bold']
        ))
        print()
        return False

    last_checkpoint_number = max(checkpoints_numbers)
    checkpoint_name = f"checkpoint-{last_checkpoint_number}"
    print(colored(
        f"Will resume from checkpoint '{checkpoint_name}'.",
        'green',
        attrs=['bold']
    ))
    print()
    resume_from_checkpoint = os.path.join(output_dir, checkpoint_name)
    return resume_from_checkpoint


early_abort: Any = False


class TrainerControlCallback(TrainerCallback):
    def on_log(self, args, state, control, **kwargs):
        global early_abort
        if os.path.isfile(os.path.join(project_dir, 'save_now')):
            print(colored(
                "'save_now' file detected! Saving a checkpoint now.",
                attrs=['bold']
            ))
            print()
            control.should_save = True
            os.remove(os.path.join(project_dir, 'save_now'))

        if os.path.isfile(os.path.join(project_dir, 'abort')):
            print(colored(
                "'abort' file detected! Stopping training now.",
                'yellow',
                attrs=['bold']
            ))
            print()
            control.should_training_stop = True
            early_abort = {
                'epoch': state.epoch,
                'global_step': state.global_step,
            }
            os.remove(os.path.join(project_dir, 'abort'))


if __name__ == "__main__":
    fire.Fire(main)
