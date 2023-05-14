from typing import Union, List

import os
import yaml

from ..formatting import (
    human_short_number as hs_number,
)

from .config_base import ConfigBase
from .tokenizer_config import TokenizerConfig
from .training_config import TrainingConfig

file_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(os.path.dirname(file_dir))
default_config_file_path = os.path.join(
    project_dir,
    'configs',
    'default.yaml'
)


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
