from typing import Union

import os

file_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(file_dir)
default_data_dir_path = os.path.join(
    project_dir,
    'data'
)


class Paths:
    def __init__(self, data_dir_path: Union[str, None]):
        if not data_dir_path:
            data_dir_path = default_data_dir_path
        os.makedirs(data_dir_path, exist_ok=True)
        self.data_dir_path = data_dir_path

    @property
    def tokenizers_path(self) -> str:
        path = os.path.join(self.data_dir_path, 'tokenizers')
        os.makedirs(path, exist_ok=True)
        return path

    def get_tokenizer_path(self, tokenizer_name) -> str:
        path = os.path.join(self.tokenizers_path, tokenizer_name)
        return path

    @property
    def datasets_path(self) -> str:
        path = os.path.join(self.data_dir_path, 'datasets')
        os.makedirs(path, exist_ok=True)
        return path

    def get_dataset_path(self, dataset_name) -> str:
        path = os.path.join(self.datasets_path, dataset_name)
        return path

    @property
    def models_path(self) -> str:
        path = os.path.join(self.data_dir_path, 'models')
        os.makedirs(path, exist_ok=True)
        return path

    def get_model_path(self, model_name) -> str:
        path = os.path.join(self.models_path, model_name)
        return path

    @property
    def configs_path(self) -> str:
        path = os.path.join(project_dir, 'configs')
        os.makedirs(path, exist_ok=True)
        return path

    def get_config_path(self, config_name) -> str:
        path = os.path.join(self.configs_path, f"{config_name}.yaml")
        return path

    @property
    def logs_path(self) -> str:
        path = os.path.join(self.data_dir_path, 'logs')
        os.makedirs(path, exist_ok=True)
        return path

    def get_log_path(self, log_name) -> str:
        path = os.path.join(self.logs_path, log_name)
        return path
