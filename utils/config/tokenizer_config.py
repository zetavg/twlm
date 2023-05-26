import json
import hashlib

from .config_base import ConfigBase
from ..data_processing import deep_sort_dict


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
        return self.get_hash(settings)
