from typing import Any, Union, List, Callable

import os
import json

from ..data_processing import (
    shallow_diff_list,
)

file_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(os.path.dirname(file_dir))
default_config_file_path = os.path.join(
    project_dir,
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
