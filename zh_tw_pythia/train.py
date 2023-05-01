import fire
import pdb

import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForSeq2Seq
)
from datasets import Dataset, load_dataset

import huggingface_hub
import wandb

import os
import re
from datetime import datetime, timezone
import traceback
import time

from typing import Union, List, Dict, Any

started_time = datetime.now(timezone.utc)
started_time_str = started_time.strftime('%Y-%m-%d-%H-%M-%S')


def main(
    base_model: str,
    tokenizer: str,
    dataset: str,
    dataset_column: str,
    dataset_split: str = "train",
    train_params: Union[str, None] = None,
    # Output
    output_dir: str = f"./output-{started_time_str}",
    run_name: Union[str, None] = None,
    # Hugging Face Hub
    push_to_hf: bool = False,
    push_to_hf_hub_model_name: Union[str, None] = None,
    hf_hub_private_repo: bool = False,
    # WandB
    wandb_project: Union[str, None] = None,
    wandb_group: Union[str, None] = None,
    wandb_tags: Union[str, None] = None,
    # Training
    per_device_train_batch_size: int = 1,
    # Logging & Saving
    logging_steps: int = 10,
    save_steps: int = 1000,
    save_total_limit: int = 10,
    # Other
    train_data_limit: Union[int, None] = None,
    **kwargs
):
    '''
    :param base_model: The tokenizer to use. Can be a path to a tokenizer or a model name on Hugging Face Hub, such as 'EleutherAI/pythia-1b', 'EleutherAI/pythia-410m', 'EleutherAI/pythia-160m' or 'EleutherAI/pythia-70m'.
    :param tokenizer: The tokenizer to use. Can be a path to a tokenizer or a model name on Hugging Face Hub, such as 'zetavg/test-pythia-zh-tw-tokenizer-20230430-1'.
    :param dataset: Dataset used for training. Can be a path to a dataset or a dataset name on Hugging Face Hub, such as 'zetavg/wikipedia_random_page_summaries_zh_tw_5k'.
    :param dataset_column: The column of the dataset to use for training, such as 'page_summary'.
    :param dataset_split: The split of the dataset to use for training, such as 'train'.

    :param train_params: If set, will only train the matching parameters of the model. Example: 'embed_in,embed_out,layers.[0-9]+.attention'.

    :param output_dir: A directory to save the checkpoints and the trained model.
    :param run_name: The name of the run. If not set, a random name will be generated. This will be used as the run name of WandB (if enabled) and the default name of the model to push to Hugging Face Hub (if enabled).

    :param push_to_hf: If enabled, the model will be pushed to Hugging Face Hub on each checkpoint save and after the training. To use this, you need to login to Hugging Face using `huggingface-cli login`; or set the HUGGING_FACE_HUB_TOKEN environment variable to your Hugging Face API token, which you can get from https://huggingface.co/settings/token. The model will be public by default, to push to a private model, use the `--hf_hub_private_repo` flag.
    :param push_to_hf_hub_model_name: If not set, a name based on the current date and time will be generated.
    :param hf_hub_private_repo: If set, the Hugging Face Hub repo will be set to private.

    :param wandb_project: The name of the project where you're sending the logs of the run. WandB will be enabled if this is set. You will need to login to WandB using `wandb login` or set the WANDB_API_KEY environment variable to your WandB API key, which can be found at https://wandb.ai/authorize.
    :param wandb_group: (optional) Specify a group to organize individual runs into a larger experiment.
    :param wandb_tags: (optional) A list tags splitted by ","", which will populate the list of tags on the run in the UI. Example: --wandb_tags='pythia-70m,wikitext'.

    :param train_data_limit: If set, will limit the number of training examples to the specified number.
    '''
    default_run_name = f"zh_tw_pythia-{started_time_str}"
    if not run_name:
        run_name = default_run_name
    if push_to_hf and not push_to_hf_hub_model_name:
        push_to_hf_hub_model_name = default_run_name

    print(f"Base model: {base_model}.")
    print(f"Tokenizer: {tokenizer}.")
    print(f"Output dir: {output_dir}.")
    run_tags = [
        f"base_model:{base_model}",
        f"tokenizer:{tokenizer}",
        f"dataset:{dataset}",
    ]

    resume_from_checkpoint = find_checkpoint_to_resume(output_dir)

    if push_to_hf_hub_model_name is not None:
        assert_is_non_blank_string(
            push_to_hf_hub_model_name, 'push_to_hf_hub_model_name')
        api = huggingface_hub.HfApi()
        try:
            user_info = api.whoami()
            print(f"Current login to HF as: {user_info['name']}.")
            print(
                f"Will push model to {'private' if hf_hub_private_repo else 'public'} repo: '{push_to_hf_hub_model_name}'.")
            if not hf_hub_private_repo:
                print("  Note: use --hf_hub_private_repo to push to a private repo.")
        except OSError as e:
            raise type(e)(
                "Please login to Hugging Face using `huggingface-cli login`; or set the HUGGING_FACE_HUB_TOKEN environment variable to your Hugging Face API token, which you can get from https://huggingface.co/settings/token."
            ) from e

    use_wandb = False
    if wandb_project is not None:
        assert_is_non_blank_string(
            wandb_project, 'wandb_project')
        use_wandb = True

    if use_wandb:
        wandb_tags_list = run_tags
        if wandb_tags:
            wandb_tags_list += [
                tag.strip() for tag in wandb_tags.split(",")]
        wandb.init(
            project=wandb_project,
            name=run_name,
            resume="allow",
            id=f"{wandb_project}--{run_name}",  # Unique ID for resuming
            group=wandb_group,
            tags=wandb_tags_list,
            save_code=True,
            magic=True,
        )

    train(
        base_model_name=base_model,
        tokenizer_name=tokenizer,
        dataset_name=dataset,
        dataset_column=dataset_column,
        dataset_split=dataset_split,
        train_data_limit=train_data_limit,
        train_params=[param_name.strip()
                      for param_name in train_params.split(",")] if train_params else None,
        per_device_train_batch_size=per_device_train_batch_size,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        output_dir_path=output_dir,
        resume_from_checkpoint=resume_from_checkpoint,
        push_to_hf_hub_model_name=push_to_hf_hub_model_name,
        hf_hub_private_repo=hf_hub_private_repo,
        use_wandb=use_wandb,
        other_args=kwargs,
    )


def train(
    base_model_name: str,
    tokenizer_name: str,
    dataset_name: str,
    dataset_column: str,
    dataset_split: str,
    train_data_limit: Union[int, None],
    train_params: Union[List[str], None],
    per_device_train_batch_size: int,
    logging_steps: int,
    save_steps: int,
    save_total_limit: int,
    output_dir_path: str,
    resume_from_checkpoint: Union[str, bool],
    push_to_hf_hub_model_name: Union[str, None],
    hf_hub_private_repo: bool,
    use_wandb: bool,
    other_args: Dict[str, Any],
):

    print(f"Loading tokenizer '{tokenizer_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # if no pad token, set it to eos
    if tokenizer.pad_token is None:
        print(
            f"Tokenizer has no pad_token set, setting it to 1.")
        tokenizer.pad_token_id = 1
    print(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}.")

    print(f"Loading base model '{base_model_name}'...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name, device_map='auto')
    print(
        f"Base model loaded. Original input_embeddings: {model.get_input_embeddings()}, output_embeddings: {model.get_output_embeddings()}.")

    print(f"Resizing model to match tokenizer vocab size...")
    model.resize_token_embeddings(tokenizer.vocab_size)
    print(
        f"New input_embeddings: {model.get_input_embeddings()}, output_embeddings: {model.get_output_embeddings()}.")

    if train_params and len(train_params) > 0:
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
        print("frozen_params:", frozen_params_list)
        print()
        if use_wandb:
            wandb.config.update({
                'train_params': train_params,
                'trainable_params': trainable_params_list,
                'frozen_params': frozen_params_list,
            })

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
    if use_wandb:
        wandb.config.update({
            'train_params': train_params,
            'trainable_params_count': trainable_params_count,
            'all_params_count': all_params_count,
            'trainable_params_rate': trainable_params_rate,
        })

    print(f"Base model ready.")

    print(f"Loading dataset '{dataset_name}'...")
    ds: Dataset = load_dataset(dataset_name)[dataset_split]  # type: ignore
    if train_data_limit:
        ds = ds.filter(lambda _, idx: idx <
                       train_data_limit, with_indices=True)

    def tokenize_data(data_point):
        batch_encoding = tokenizer(
            # See: https://huggingface.co/docs/transformers/main/en/main_classes/tokenizer#tokenizer
            data_point[dataset_column],
            max_length=1024,
            truncation=True,
            padding=False,  # Handled by DataCollatorForSeq2Seq.
            return_tensors=None  # Handled by the trainer.
        )
        batch_encoding["labels"] = batch_encoding["input_ids"].copy()
        # This is handled by the trainer.
        # batch_encoding = {k: v.to(device) for k, v in batch_encoding.items()}
        return batch_encoding

    train_data = ds.map(tokenize_data)
    train_data = train_data.filter(lambda x: len(x["input_ids"]) > 0)
    # print("Sample train_data:", train_data[0])

    data_collator = DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    )

    # See: https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments
    training_args = TrainingArguments(
        # Output dir
        output_dir=output_dir_path,
        overwrite_output_dir=True,
        # Train hyperparams
        num_train_epochs=1,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=1,
        optim="adamw_torch",
        learning_rate=5e-5,
        lr_scheduler_type="constant",
        warmup_steps=100,
        # Steps
        logging_steps=logging_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        # HF
        push_to_hub=True if push_to_hf_hub_model_name else False,
        hub_model_id=push_to_hf_hub_model_name,
        hub_private_repo=hf_hub_private_repo,
        # hub_strategy="all_checkpoints",  # Problematic
        hub_strategy="end",
        # WandB
        report_to=['wandb'] if use_wandb else None,
        # Other
        # fp16=True
        **other_args
    )

    # See: https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.Trainer
    trainer = TrainerWithOutputLogging(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_data,  # type: ignore
        data_collator=data_collator,
        args=training_args
    )
    trainer._output_logging_tokenizer = tokenizer  # type: ignore
    trainer.log_output_every_n_steps = logging_steps * 50  # type: ignore

    if resume_from_checkpoint:
        if isinstance(resume_from_checkpoint, str):
            print(f"Resuming from checkpoint '{resume_from_checkpoint}'...")
        else:
            print(f"Resuming from latest checkpoint...")

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir_path)
    print(f"Model saved to {output_dir_path}.")
    tokenizer.save_pretrained(output_dir_path)
    print(f"Tokenizer saved to {output_dir_path}.")
    if push_to_hf_hub_model_name is not None:
        print("Pushing model to HF Hub...")
        results = model.push_to_hub(
            push_to_hf_hub_model_name,
            private=hf_hub_private_repo)
        print(results)
        print("Pushing tokenizer to HF Hub...")
        results = tokenizer.push_to_hub(
            push_to_hf_hub_model_name,
            private=hf_hub_private_repo)
        print(results)
    print('Done.')


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
                output_text = tokenizer.decode(token_ids)
                labels = inputs['labels'][0]
                labels_to_decode = labels[1:]
                labels_to_decode = labels_to_decode[labels_to_decode > 0]
                label_text = tokenizer.decode(labels_to_decode)

                self.log({
                    'output_text': output_text,
                    'label_text': label_text,
                    'output_token_ids': token_ids,
                    'labels': labels.tolist()
                })
                print("output:", output_text)
                print(" label:", label_text)
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
        print(
            "Non of the checkpoints are valid, will not resume from checkpoint.")
        return False

    # Find the latest checkpoint.
    checkpoints_number_matches = [
        re.search(r'checkpoint-(\d+)$', ckpt)
        for ckpt in filtered_checkpoints]
    checkpoints_numbers = [int(m.group(1))
                           for m in checkpoints_number_matches if m]
    if len(checkpoints_numbers) <= 0:
        print(
            "Non of the checkpoints are valid, will not resume from checkpoint.")
        return False

    last_checkpoint_number = max(checkpoints_numbers)
    checkpoint_name = f"checkpoint-{last_checkpoint_number}"
    print(f"Will resume from checkpoint '{checkpoint_name}'.")
    resume_from_checkpoint = os.path.join(output_dir, checkpoint_name)
    return resume_from_checkpoint


def assert_is_non_blank_string(s, string_name):
    assert isinstance(
        s, str), f"A string value must be provided for {string_name}!"
    assert isinstance(
        s, str), f"A string value must be provided for {string_name}!"


if __name__ == "__main__":
    fire.Fire(main)
