# Taiwanese Mandarin LM

An attempt to re-train EN language models to understand and generate fluent Taiwanese Mandarin (Traditional Chinese).

## Trained Models

* [TW-Pythia-6.9B-Chat](https://huggingface.co/twlm/tw-pythia-6.9b-chat-v0_2)

Demo: see https://hackmd.io/@z/twlm-demo

## Usage

主要有三個步驟：

1. Build Tokenizer - 擴充指定的 base tokenizer 加入新的中文 token (`python build_tokenizer.py`)。
2. Prepare Dataset - 準備訓練資料集 (`python prepare_dataset.py <train_name>`)。
3. Train Model - 訓練模型 (`python train.py <train_name>`)。

每個步驟的參數細節都由 config 檔決定，詳細請參考 `configs/sample.yaml` 的內容。

Train 可以定義多個，每個可以使用不同的訓練資料、超參數，以及可訓練的參數。例如，可以在 config 檔中這寫：

```yaml
training:
  # 第一次訓練，只訓練 text embedding
  embeddings:
    dataset:
      # ...
    only_train_parameters_matching:
      - 'embed'  # 只訓練名字符合 'embed' 的參數
    training_arguments:
      # ...

  # 第二次訓練，做 instruction tuning
  instruction_tuning:
    base_on:
      output_of: embeddings  # 基於 'embeddings' 訓練後產出的模型繼續訓練
    dataset:
      # ...
    training_arguments:
      # ...
```

依照以上定義的參數，可以執行 `python prepare_dataset.py embeddings` 來準備 'embeddings' 的訓練資料，然後執行 `python train.py embeddings` 開始 'embeddings' 的訓練。

以上指令都可以用 `--cfg` 來指定要使用哪一個 config 檔，例如 `python build_tokenizer.py --cfg default` 為使用 `configs/default.yaml`。亦可以用 `--config_file_path` 來指定 config 檔的路徑，例如 `python train.py --config_file_path '~/configs/80k_tokens.yaml`。


### 立即存擋以及提前中止

在 `train.py` 執行訓練的過程中，若偵測到專案目錄中存在名為 `save_now` 檔案，將會立即儲存一份 checkpoint。

而若偵測到專案目錄中存在名為 `abort` 的檔案，將會提前中止訓練。提前終止的訓練仍然會儲存 model，以及將 model 上傳到 Hugging Face Hub（若有啟用）。提前終止而上傳到 Hugging Face Hub 的模型將會在 model card 上自動註記提前終止時的 epoch 及 global step。

舉例來說，我們可以切換到 train.py 所在的目錄下，執行 `touch save_now` 來立即存檔，或執行 `touch abort` 提前中止訓練。


### 使用 SkyPilot 在雲端訓練

（需要先安裝以及設定好 SkyPilot，詳見： https://skypilot.readthedocs.io/en/latest/getting-started/installation.html 。）

首先，將 `sky_training.yaml.sample` 檔案複製為 `sky_training.yaml` (`cp sky_training.yaml.sample sky_training.yaml`)，然後編輯 `sky_training.yaml` 的內容，調整要使用的機器資源以及 storage bucket。

接著，若有需要，複製 `sky_prepare.sh.sample` 檔案為 `sky_prepare.sh`，並編輯其內容，加入要在每次開始雲端訓練前執行的指令，例如切換到特定的 Google Cloud 設定檔。

準備完成後，執行 `./sky_train.sh <train_name>`，即可開始雲端訓練。`sky_train.sh` 封裝了原本的 `sky launch` 或 `sky exec` 指令，會將本地端必要的訓練程式碼複製到雲端，同時將本機已登入的 Hugging Face 與 Weights & Biases 認證資訊與雲端機器共享。

使用 `./sky_train.sh` 與 `python train.py` 相同，可以使用 `--cfg` 來指定要使用的 config（但不支援 `--config_file_path`）。

除此之外，`./sky_train.sh` 還可以使用 `--cluster_name <name>` 或是 `-n <name>` 來指定要使用的 SkyPilot cluster (等同 `sky launch` 的 `-c`)，以及使用 `--skip_setup` 或 `-s` 來跳過雲端機器的 setup (若使用了 `--skip_setup`，背後將會使用 `sky exec` 而非 `sky launch`)。


### 其他工具

* 預覽 dataset：`python preview_dataset.py --cfg=... <train_name> --split=test --range_=10,20` (參數基本與 `train.py` 相同，但多了 `--split`、`--range_` 以及 `--only_preview` 三個參數)。
* 訓練前初步檢查 config 內容：`python train_check_config.py --cfg=... <train_name>`。
* 比較兩份 config 的差異：`python diff_configs.py config_1 config_2`。

## Related Projects

* [zetavg/LLM-Research](https://github.com/zetavg/LLM-Research)
* [zetavg/LLaMA-LoRA-Tuner](https://github.com/zetavg/LLaMA-LoRA-Tuner)
