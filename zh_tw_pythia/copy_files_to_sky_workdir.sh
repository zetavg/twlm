#!/usr/bin/env bash

set -e

SCRIPT_DIR="$(dirname "$(realpath "$0")")"

mkdir -p "${SCRIPT_DIR}/sky_workdir"

cp -f "${SCRIPT_DIR}/requirements.txt" "${SCRIPT_DIR}/sky_workdir/"
cp -rf "${SCRIPT_DIR}/configs" "${SCRIPT_DIR}/sky_workdir/"
cp -rf "${SCRIPT_DIR}/utils" "${SCRIPT_DIR}/sky_workdir/"
cp -f "${SCRIPT_DIR}/build_tokenizer.py" "${SCRIPT_DIR}/sky_workdir/"
cp -f "${SCRIPT_DIR}/prepare_dataset.py" "${SCRIPT_DIR}/sky_workdir/"
cp -f "${SCRIPT_DIR}/train.py" "${SCRIPT_DIR}/sky_workdir/"
cp -f "${SCRIPT_DIR}/train_check_config.py" "${SCRIPT_DIR}/sky_workdir/"
cp -f "${SCRIPT_DIR}/merge_lora.py" "${SCRIPT_DIR}/sky_workdir/"
cp -f "${SCRIPT_DIR}/diff_configs.py" "${SCRIPT_DIR}/sky_workdir/"

echo "Files copied to sky_workdir."
