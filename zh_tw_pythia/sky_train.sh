#!/usr/bin/env bash


# Initialize default values
cfg="default"
cluster_name="zh-tw-model-trainer"
skip_setup="false"

# Usage function
usage() {
  echo "Usage: $0 <train_name> [-c|--cfg CONFIG_NAME] [-n|--cluster_name CLUSTER_NAME] [-s|--skip_setup]"
  echo
  echo "Arguments:"
  echo "  <train_name>        The train to run, required."
  echo "  -c, --cfg           Config name to use, defaults to '$cfg'."
  echo "  -n, --cluster_name  Cluster name to use, defaults to '$cluster_name'."
  echo "  --skip_setup        Skips setup, defaults to $skip_setup."
  echo
  echo "  --help              Display this help message and exit."
}

# Check if no arguments are provided
if [ $# -eq 0 ]; then
  echo "Error: No arguments provided."
  usage
  exit 1
fi

# Parse arguments
while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
    -c|--cfg)
      cfg="$2"
      shift # past argument
      shift # past value
      ;;
    -c=*|--cfg=*)
      cfg="${1#*=}"
      shift # past argument
      ;;
    -n|--cluster_name)
      cluster_name="$2"
      shift # past argument
      shift # past value
      ;;
    -n=*|--cluster_name=*)
      cluster_name="${1#*=}"
      shift # past argument
      ;;
    -s|--skip_setup)
      skip_setup="true"
      shift # past argument
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      if [[ -z $train_name ]]; then
        train_name="$1"
        shift # past argument
      else
        echo "Error: Unknown option '$1'"
        usage
        exit 1
      fi
      ;;
  esac
done

# Check if train_name is provided
if [[ -z $train_name ]]; then
  echo "Error: train_name not provided."
  usage
  exit 1
fi
# Run the script with the provided arguments
BASEDIR=$(cd "$(dirname "$0")"; pwd)

cd $BASEDIR

python train_check_config.py "$train_name" --cfg="$cfg"

echo "Using cluster name: $cluster_name"
echo "Skip setup: $skip_setup"
echo ""

if [ -f "$BASEDIR/sky_prepare.sh" ]; then
  echo "Running sky_prepare.sh"
  bash "$BASEDIR/sky_prepare.sh"
  echo ""
fi

./copy_files_to_sky_workdir.sh
echo ""

if [ "$skip_setup" == "true" ]; then
  sky exec "$cluster_name" sky_training.yaml \
    --env WANDB_API_KEY="$(awk -v machine="api.wandb.ai" 'BEGIN {RS="\n"; FS="\n"} $1 == "machine " machine {getline; while ($0 != "" && $0 !~ /^machine/) {if ($0 ~ /^ *password/) {sub(/^ *password */, "", $0); print $0; exit}; getline}}' ~/.netrc)" \
    --env HUGGING_FACE_HUB_TOKEN="$(cat ~/.cache/huggingface/token | tr -d '\n')" \
    --env CFG="$cfg" \
    --env TRAIN_NAME="$train_name"

else
  sleep 8
  sky launch sky_training.yaml \
    --env WANDB_API_KEY="$(awk -v machine="api.wandb.ai" 'BEGIN {RS="\n"; FS="\n"} $1 == "machine " machine {getline; while ($0 != "" && $0 !~ /^machine/) {if ($0 ~ /^ *password/) {sub(/^ *password */, "", $0); print $0; exit}; getline}}' ~/.netrc)" \
    --env HUGGING_FACE_HUB_TOKEN="$(cat ~/.cache/huggingface/token | tr -d '\n')" \
    --env CFG="$cfg" \
    --env TRAIN_NAME="$train_name" \
    -c "$cluster_name" \
    --retry-until-up \
    -y
fi
