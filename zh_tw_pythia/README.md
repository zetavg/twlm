# Traditional Chinese (Taiwan) Pythia

An attempt to train a Pythia model to understand and speak fluent Taiwan Traditional Chinese.

## Train on the Cloud with SkyPilot

Modify `sky_task.yaml` and run it like:

```bash
./copy_files_to_sky_workdir && sky launch sky_task.yaml --env WANDB_API_KEY="$(awk -v machine="api.wandb.ai" 'BEGIN {RS="\n"; FS="\n"} $1 == "machine " machine {getline; while ($0 != "" && $0 !~ /^machine/) {if ($0 ~ /^ *password/) {sub(/^ *password */, "", $0); print $0; exit}; getline}}' ~/.netrc)" --env HUGGING_FACE_HUB_TOKEN="$(cat ~/.cache/huggingface/token | tr -d '\n')" -c zh-tw-model-trainer
```

Or use [SkyPilot Managed Spot Jobs](https://skypilot.readthedocs.io/en/latest/examples/spot-jobs.html):

```bash
./copy_files_to_sky_workdir && sky spot launch sky_task.yaml --env WANDB_API_KEY="$(awk -v machine="api.wandb.ai" 'BEGIN {RS="\n"; FS="\n"} $1 == "machine " machine {getline; while ($0 != "" && $0 !~ /^machine/) {if ($0 ~ /^ *password/) {sub(/^ *password */, "", $0); print $0; exit}; getline}}' ~/.netrc)" --env HUGGING_FACE_HUB_TOKEN="$(cat ~/.cache/huggingface/token | tr -d '\n')" -n zh-tw-model-training
```
