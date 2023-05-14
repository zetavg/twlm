import json
import hashlib

from .config_base import ConfigBase


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
