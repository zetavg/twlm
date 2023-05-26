from typing import Any, Union, List

import re
import json
import hashlib

from ..formatting import (
    remove_common_substring,
)

from .config_base import ConfigBase


class TrainingDatasetConfig(ConfigBase):
    ''' Config for train dataset. '''

    @property
    def max_tokens_length(self) -> int:
        assert self.parent_config, "parent_config is not passed"
        return self.parent_config.max_tokens_length

    @property
    def preview_length(self) -> int:
        return self._get_value('preview_length', int)

    @property
    def sort_by(self):
        sort_by_str = self._get_value('sort_by', str, allow_none=True)
        if sort_by_str:
            column, order = sort_by_str.split('-')
            return column, order
        return None

    @property
    def test_size(self) -> Union[int, float, None]:
        return self._get_value('test_size', Union[int, float], allow_none=True)

    @property
    def train_test_split_seed(self) -> Union[int, None]:
        return self._get_value('train_test_split_seed', int, allow_none=True)

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
            tokenizer_name = \
                remove_common_substring(tokenizer_name, project_and_group_name)
            tokenizer_name = tokenizer_name.strip('-_')

            generated_name = f"{project_and_group_name}-{tokenizer_name}"

            generated_name += f"-{self.parent_config.config_name}"

            generated_name += f"-{self.get_build_with_short_str()}"
            generated_name += f"-{self.get_settings_hash()[:6]}"

            generated_name += f"-c{self.max_tokens_length}"

            if len(generated_name) > 80:
                raise ValueError(f"Auto generated dataset_name is too long, it must be less than 80 chars. ('{generated_name}')")

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
        if name == 'wikipedia':
            return 'wiki'
        if name == 'sharegpt':
            return 'sg'
        elif name == 'alpaca':
            return 'alp'
        return ''

    def get_settings_hash_of(self, name) -> str:
        settings = self.get_settings_for(name)
        if name == 'translations':
            source_dataset = settings.get('source_dataset') or ''
            assert isinstance(
                source_dataset, str), "source_dataset must be str"
            return hashlib.sha256(source_dataset.encode('utf-8')).hexdigest()

        return self.get_hash(settings)

    def get_settings_hash(self) -> str:
        all_hash = ''
        for n in self.build_with:
            all_hash += self.get_settings_hash_of(n)
        return hashlib.sha256(all_hash.encode('utf-8')).hexdigest()


class TrainingConfig(ConfigBase):
    ''' Config for training. '''

    @property
    def use_peft(self) -> Union[str, None]:
        return self._get_value('use_peft', str, allow_none=True)

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
    def base_on_model_name(self) -> str:
        assert self.parent_config, "parent_config is not passed"
        model_name = self.parent_config.base_model_name

        base_on = self.base_on
        if base_on:
            if base_on.get('model'):
                model_name = base_on['model']
            elif base_on.get('output_of'):
                based_t_cfg = self.parent_config.get_training_config(base_on['output_of'])
                model_name = based_t_cfg.model_name
            else:
                raise ValueError(f"'{base_on}' is not a valid 'base_on' value.")

        return model_name

    @property
    def max_tokens_length(self) -> int:
        return self._get_value('max_tokens_length', int)

    @property
    def dataset_config(self) -> TrainingDatasetConfig:
        def get_new_training_dataset_config():
            dataset_config = self._get_value('dataset', dict)
            if not dataset_config:
                raise ValueError(
                    f"Missing training dataset config {'.'.join(self.config_level)}.dataset in {self.config_file_path}.")

            if dataset_config.get('same_as'):
                assert self.parent_config, "parent_config is not passed"
                root_config = self.parent_config

                t_config = root_config.get_training_config(
                    dataset_config.get('same_as'))

                return t_config.dataset_config

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

            # project_name = root_config.project_name
            group_name = root_config.group_name
            # project_and_group_name = f"{project_name}-{group_name}"

            base_model_name = root_config.base_model_name
            base_model_name = re.sub(r'^[^/]+/', '', base_model_name)
            base_model_name = \
                remove_common_substring(base_model_name, group_name)
            base_model_name = base_model_name.strip('-_')

            tokenizer_name = root_config.tokenizer_name
            tokenizer_name = tokenizer_name.replace('-tokenizer-a', '-ta')
            tokenizer_name = \
                remove_common_substring(tokenizer_name, group_name)
            tokenizer_name = \
                remove_common_substring(tokenizer_name, base_model_name)
            tokenizer_name = tokenizer_name.strip('-_')

            train_name = self.config_name
            run_name_suffix = \
                (self._get_value('run_name_suffix', str, allow_none=True) or
                 self._get_value('run_suffix', str, allow_none=True))

            generated_name = f"{group_name}-{train_name}-{tokenizer_name}-{base_model_name}"
            if run_name_suffix:
                generated_name += f"-{run_name_suffix}"
            generated_name += f"-{self.hash[:6]}"

            generated_name = generated_name.strip('-_')

            if len(generated_name) > 80:
                raise ValueError(f"Auto generated run_name is too long, it must be less than 80 chars. ('{generated_name}')")

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
            group_name = root_config.group_name
            project_and_group_name = f"{project_name}-{group_name}"

            base_model_name = root_config.base_model_name
            base_model_name = re.sub(r'^[^/]+/', '', base_model_name)
            base_model_name = \
                remove_common_substring(
                    base_model_name, project_and_group_name)
            base_model_name = base_model_name.strip('-_')

            tokenizer_name = root_config.tokenizer_name
            tokenizer_name = tokenizer_name.replace('-tokenizer-a', '-ta')
            tokenizer_name = \
                remove_common_substring(tokenizer_name, project_and_group_name)
            tokenizer_name = \
                remove_common_substring(tokenizer_name, base_model_name)
            tokenizer_name = tokenizer_name.strip('-_')

            train_name = self.config_name
            run_name_suffix = \
                (self._get_value('run_name_suffix', str, allow_none=True) or
                 self._get_value('run_suffix', str, allow_none=True))

            generated_name = f"{project_and_group_name}-{base_model_name}-{tokenizer_name}-{train_name}"
            if run_name_suffix:
                generated_name += f"-{run_name_suffix}"
            generated_name += f"-{self.hash[:6]}"

            generated_name = generated_name.strip('-_')

            if len(generated_name) > 80:
                raise ValueError(f"Auto generated model_name is too long, it must be less than 80 chars. ('{generated_name}')")

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
        config = {
            k: v
            for k, v in self._config.items()
            if k != 'base_on'
            and k != 'dataset'
            and k != 'run_name_suffix'
            and k != 'log_output_every_n_steps'
        }
        config['dataset'] = self.dataset_name
        config['base_on_model_name'] = self.base_on_model_name
        if config.get('training_arguments'):
            config['training_arguments'] = {
                k: v
                for k, v in config['training_arguments'].items()
                if k not in self.training_argument_keys_allow_updating
            }
        # print(json.dumps(config, indent=2))
        return self.get_hash(config)
