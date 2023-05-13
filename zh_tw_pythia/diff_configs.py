from typing import Union

import os
import fire
import json

from utils.config import Config
from utils.paths import Paths
from utils.data_processing import deep_diff_dict


def main(
    config_1_name_or_path: str,
    config_2_name_or_path: str,
):
    paths = Paths(None)

    config_1_file_path = config_1_name_or_path
    if not os.path.isfile(config_1_file_path):
        config_1_file_path = paths.get_config_path(config_1_file_path)

    config_2_file_path = config_2_name_or_path
    if not os.path.isfile(config_2_file_path):
        config_2_file_path = paths.get_config_path(config_2_file_path)

    config_1 = Config(config_1_file_path)
    config_2 = Config(config_2_file_path)

    diff = deep_diff_dict(config_1._config, config_2._config)

    print(json.dumps(diff, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    fire.Fire(main)
