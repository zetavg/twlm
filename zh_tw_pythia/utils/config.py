import pdb
from typing import Any, Union, List, Callable

import os
import re
import yaml
import json
import hashlib
from datetime import datetime, timezone

from .data_processing import (
    shallow_diff_list,
)
from .formatting import (
    human_short_number as hs_number,
    remove_common_substring,
)

file_dir = os.path.dirname(os.path.abspath(__file__))
default_config_file_path = os.path.join(
    os.path.dirname(file_dir),
    'configs',
    'default.yaml'
)


class ConfigBase:
    def __init__(
            self,
            config: dict, config_file_path: str, config_level: list = [],
            parent_config=None, config_name=None):
        self._config = config
        self.config_file_path = config_file_path
        self._generated_values = {}
        self.config_level = config_level
        self.parent_config = parent_config
        self.config_name = config_name

    def get_config_level_str(self, more: List[str] = []) -> str:
        config_level = self.config_level + more
        return '.'.join(config_level)

    def _get_value(
        self,
        key: str,
        type: type,
        allow_none: bool = False,
        allowed_keys: Union[None, List[str]] = None,
    ) -> Any:
        val = self._config.get(key)
        if not val:
            if not allow_none:
                k_path = '.'.join(self.config_level + [key])
                raise ValueError(
                    f"Please specify '{k_path}' in {self.config_file_path}.")
            return val
        if not isinstance(val, type):
            k_path = '.'.join(self.config_level + [key])
            raise ValueError(
                f"'{k_path}' in {self.config_file_path} must be {type}. Check {self.config_file_path}.")

        if allowed_keys:
            diff_result = shallow_diff_list(val.keys(), allowed_keys)
            value_path = self.get_config_level_str([key])
            if diff_result['added']:
                raise ValueError(
                    f"Unknown keys {diff_result['added']} for {value_path} in {self.config_file_path}. Known keys are: {', '.join(allowed_keys)}.")
        return val

    def _get_cached(self, name: str, generator: Callable[[], Any]):
        if self._generated_values.get(name):
            return self._generated_values[name]
        val = generator()
        self._generated_values[name] = val
        # print(f"Generated {name}: {val}")
        return val

    def to_json(self, indent=None):
        return json.dumps(self._config, indent=indent, ensure_ascii=False)

    # def _get_ds_sc(
    #     self,
    # ) -> str:
    #     t = datetime.now(timezone.utc)
    #     t_str = t.strftime('%Y%m%d')
    #     return t_str


class TokenizerConfig(ConfigBase):
    ''' Config for building the tokenizer. '''

    @property
    def build_with(self) -> str:
        return self._get_value('build_with', str)

    @property
    def tokens_to_add(self) -> int:
        return self._get_value('tokens_to_add', int)

    @property
    def settings(self) -> dict:
        build_with = self._get_value('build_with', str)
        return self._get_value(f'{build_with}_settings', dict)

    @property
    def settings_name(self) -> str:
        build_with = self._get_value('build_with', str)
        return f'{build_with}_settings'

    @property
    def settings_hash(self) -> str:
        settings = self.settings
        sorted_items = sorted(settings.items(), key=lambda x: x[0])
        sorted_tuple = tuple(sorted_items)
        sorted_json = json.dumps(sorted_tuple, sort_keys=True).encode('utf-8')
        return hashlib.sha256(sorted_json).hexdigest()


class TrainingDatasetConfig(ConfigBase):
    ''' Config for train dataset. '''

    @property
    def max_training_text_length(self) -> int:
        assert self.parent_config, "parent_config is not passed"
        return self.parent_config.max_training_text_length

    @property
    def preview_length(self) -> int:
        return self._get_value('preview_length', int)

    @property
    def dataset_name(self) -> str:
        ''' Name of the newly builded dataset for training. '''
        assert self.parent_config, "parent_config is not passed"
        dataset_name = self.parent_config._config.get('dataset_name')
        if dataset_name:
            return dataset_name

        def get_generated_name():
            assert self.parent_config, "parent_config is not passed"
            assert self.parent_config.parent_config, "cannot access self.parent_config.parent_config"

            root_config = self.parent_config.parent_config

            project_name = root_config.project_name
            group_name = root_config.group_name
            project_and_group_name = f"{project_name}-{group_name}"
            tokenizer_name = root_config.tokenizer_name
            tokenizer_name = tokenizer_name.replace('-tokenizer-a', '-ta')

            generated_name = f"{project_name}-{group_name}-{tokenizer_name}"
            if tokenizer_name.startswith(project_and_group_name):
                generated_name = tokenizer_name
            generated_name += f"-{self.parent_config.config_name}"

            generated_name += f"-{self.get_build_with_short_str()}"
            generated_name += f"-{self.get_settings_hash()[:6]}"

            generated_name += f"-c{self.max_training_text_length}"

            return generated_name

        return self._get_cached(
            'dataset_name',
            get_generated_name
        )

    @property
    def build_with(self) -> List[str]:
        val = self._config.get('build_with')
        if not isinstance(val, list):
            val = [val]
        if not all(isinstance(element, str) for element in val):
            k_path = '.'.join(self.config_level + ['build_with'])
            raise ValueError(
                f"'{k_path}' in {self.config_file_path} must be a list of str. Check {self.config_file_path}.")
        return val  # type: ignore

    def get_settings_for(self, name) -> dict:
        return self._get_value(f'{name}_settings', dict)

    @property
    def settings_name(self) -> str:
        build_with = self._get_value('build_with', str)
        return f'{build_with}_settings'

    def get_build_with_short_str(self) -> str:
        return '_'.join([self.get_short_name(n) for n in self.build_with])

    def get_short_name(self, name) -> str:
        if name == 'translations':
            return 'tr'
        return ''

    def get_settings_hash_of(self, name) -> str:
        settings = self.get_settings_for(name)
        if name == 'translations':
            source_dataset = settings.get('source_dataset') or ''
            assert isinstance(
                source_dataset, str), "source_dataset must be str"
            return hashlib.sha256(source_dataset.encode('utf-8')).hexdigest()

        sorted_items = sorted(settings.items(), key=lambda x: x[0])
        sorted_tuple = tuple(sorted_items)
        sorted_json = json.dumps(sorted_tuple, sort_keys=True).encode('utf-8')
        return hashlib.sha256(sorted_json).hexdigest()

    def get_settings_hash(self) -> str:
        all_hash = ''
        for n in self.build_with:
            all_hash += self.get_settings_hash_of(n)
        return hashlib.sha256(all_hash.encode('utf-8')).hexdigest()


class TrainingConfig(ConfigBase):
    ''' Config for training. '''

    @property
    def base_on(self) -> Union[dict, None]:
        value = self._get_value('base_on', dict,
                                allow_none=True,
                                allowed_keys=['output_of', 'model'])
        if value and value.get('output_of'):
            if value['output_of'] == self.config_name:
                raise ValueError(
                    f"Invalid value '{value['output_of']}' for {self.get_config_level_str(['base_on', 'output_of'])} - it's referring to itself. Check {self.config_file_path}.")
            assert self.parent_config, "parent_config is not passed"
            try:
                self.parent_config.get_training_config(value['output_of'])
            except Exception as e:
                raise ValueError(
                    f"The train '{value['output_of']}' referenced from {self.get_config_level_str(['base_on', 'output_of'])} is not defined. Defined training: {', '.join(self.parent_config.training_names)}. Check {self.config_file_path}.") from e
        return value

    @property
    def max_training_text_length(self) -> int:
        return self._get_value('max_training_text_length', int)

    @property
    def dataset_config(self) -> TrainingDatasetConfig:
        def get_new_training_dataset_config():
            dataset_config = self._get_value('dataset', dict)
            if not dataset_config:
                raise ValueError(
                    f"Missing training dataset config {'.'.join(self.config_level)}.dataset in {self.config_file_path}.")
            return TrainingDatasetConfig(
                dataset_config,
                config_file_path=self.config_file_path,
                config_level=self.config_level + ['dataset'],
                parent_config=self)
        return self._get_cached(
            "dataset_config",
            get_new_training_dataset_config
        )

    @property
    def dataset_name(self) -> str:
        return self.dataset_config.dataset_name

    @property
    def run_name(self) -> str:
        val = self._get_value('run_name', str, allow_none=True)
        if val:
            return val

        def get_generated_name():
            assert self.parent_config, "parent_config is not passed"

            root_config = self.parent_config

            project_name = root_config.project_name
            group_name = root_config.group_name
            project_and_group_name = f"{project_name}-{group_name}"

            base_model_name = root_config.base_model_name
            base_model_name = re.sub(r'^[^/]+/', '', base_model_name)

            tokenizer_name = root_config.tokenizer_name
            tokenizer_name = tokenizer_name.replace('-tokenizer-a', '-ta')
            tokenizer_name = \
                remove_common_substring(tokenizer_name, project_and_group_name)
            tokenizer_name = \
                remove_common_substring(tokenizer_name, base_model_name)
            tokenizer_name = tokenizer_name.strip('-_')

            train_name = self.config_name
            run_suffix = self._get_value('run_suffix', str, allow_none=True)

            generated_name = f"{base_model_name}-{tokenizer_name}-{train_name}"
            if run_suffix:
                generated_name += f"-{run_suffix}"
            generated_name += f"-{self.hash[:4]}"

            generated_name = generated_name.strip('-_')

            return generated_name

        return self._get_cached(
            'run_name',
            get_generated_name
        )

    @property
    def model_name(self) -> str:
        val = self._get_value('model_name', str, allow_none=True)
        if val:
            return val

        def get_generated_name():
            assert self.parent_config, "parent_config is not passed"
            root_config = self.parent_config

            project_name = root_config.project_name
            run_name = self.run_name
            run_name = \
                remove_common_substring(run_name, project_name)
            run_name = run_name.strip('-_')

            generated_name = f"{project_name}-{run_name}"
            generated_name = generated_name.strip('-_')

            return generated_name

        return self._get_cached(
            'model_name',
            get_generated_name
        )

    @property
    def only_train_parameters_matching(self) -> Union[list, None]:
        return self._get_value(
            'only_train_parameters_matching', list, allow_none=True)

    @property
    def training_arguments(self) -> dict:
        return self._get_value('training_arguments', dict)

    @property
    def training_argument_keys_allow_updating(self) -> List[str]:
        ''' Keys that are considered safe to change during the run. '''
        return [
            'per_device_train_batch_size',
            'per_device_eval_batch_size',
            'gradient_accumulation_steps',
            'eval_accumulation_steps',
            'eval_delay',
            'log_level',
            'log_level_replica',
            'log_on_each_node',
            'logging_dir',
            'logging_strategy',
            'logging_first_step',
            'logging_steps',
            'logging_nan_inf_filter',
            'save_strategy',
            'save_steps',
            'save_total_limit',
            'save_safetensors',
            'save_on_each_node',
            'eval_steps',
        ]

    @property
    def hash(self) -> str:
        config = self._config.copy()
        if config.get('training_arguments'):
            config['training_arguments'] = {
                k: v
                for k, v in config['training_arguments'].items()
                if k not in self.training_argument_keys_allow_updating
            }
        sorted_items = sorted(config.items(), key=lambda x: x[0])
        sorted_tuple = tuple(sorted_items)
        sorted_json = json.dumps(sorted_tuple, sort_keys=True).encode('utf-8')
        return hashlib.sha256(sorted_json).hexdigest()


class Config(ConfigBase):
    ''' Config for the whole project. '''

    def __init__(self, config_file_path: Union[str, None]):
        if not config_file_path:
            config_file_path = default_config_file_path

        with open(config_file_path, 'r') as file:
            config = yaml.safe_load(file)
        self._config = config
        self.config_file_path = config_file_path
        self._generated_values = {}

        tokenizer_config = self._get_value('tokenizer', dict)
        self._tokenizer_config = TokenizerConfig(
            tokenizer_config,
            config_file_path=config_file_path,
            config_level=['tokenizer'],
            parent_config=self)

    @property
    def project_name(self) -> str:
        ''' Project name, used for logging and to name the outputs. '''
        return self._config.get('project_name') or 'zh-tw-llm'

    @property
    def group_name(self) -> str:
        ''' Group name, used for logging and to name the outputs. '''
        return self._config.get('group_name') or 'default'

    @property
    def base_tokenizer_name(self) -> str:
        return self._get_value('base_tokenizer_name', str)

    @property
    def base_model_name(self) -> str:
        return self._get_value('base_model_name', str)

    @property
    def tokenizer_name(self) -> str:
        ''' Name of the newly builded tokenizer. '''
        tokenizer_name = self._get_value(
            'tokenizer_name', str, allow_none=True)
        if tokenizer_name:
            return tokenizer_name

        t_config = self.tokenizer_config
        generated_name = f"{self.project_name}-{self.group_name}"
        generated_name += f"-tokenizer-a{hs_number(t_config.tokens_to_add)}"
        generated_name += f"-{self.tokenizer_config.settings_hash[:6]}"
        return self._get_cached(
            'tokenizer_name',
            lambda: generated_name
        )

    @property
    def tokenizer_config(self) -> TokenizerConfig:
        return self._tokenizer_config

    @property
    def training_names(self) -> List[str]:
        training_dict = self._get_value('training', dict)
        return training_dict.keys()

    def get_training_config(self, name) -> TrainingConfig:
        def get_new_training_config():
            training_dict = self._get_value('training', dict)
            training_config = training_dict.get(name)
            if not training_config:
                raise ValueError(
                    f"No such training config 'training.{name}' in {self.config_file_path}.")
            return TrainingConfig(
                training_config,
                config_file_path=self.config_file_path,
                config_level=['training', name],
                parent_config=self, config_name=name)

        return self._get_cached(
            f"training_config_{name}",
            get_new_training_config
        )

    @property
    def hf_user_or_org_name(self) -> str:
        return self._get_value('hf_user_or_org_name', str)

    @property
    def push_outputs_to_hf(self) -> bool:
        return self._get_value('push_outputs_to_hf', bool)

    @property
    def report_to_wandb(self) -> bool:
        return self._get_value('report_to_wandb', bool, allow_none=True)

    @property
    def wandb_project(self) -> str:
        val = self._get_value('wandb_project', str, allow_none=True)
        return val or self.project_name

    @property
    def wandb_group(self) -> str:
        val = self._get_value('wandb_group', str, allow_none=True)
        return val or f"{self.project_name}-{self.group_name}"
