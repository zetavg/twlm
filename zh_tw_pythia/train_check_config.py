from typing import Union

import fire

from utils.config import Config
from utils.paths import Paths
from utils.get_training_config_values import get_training_config_values


def main(
    train_name: str,
    cfg: Union[str, None] = None,
    config_file_path: Union[str, None] = None,
    data_dir_path: Union[str, None] = None,
):
    paths = Paths(data_dir_path)
    if cfg and not config_file_path:
        config_file_path = paths.get_config_path(cfg)
    config = Config(config_file_path)

    training_config = config.get_training_config(train_name)

    get_training_config_values(config, training_config, print_values=True)


if __name__ == "__main__":
    fire.Fire(main)
