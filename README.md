test_hf_dataset

How to create and upload a minimal audio dataset to HuggingFace Hub.

- [References](#references)
- [Installation](#installation)
- [Add HuggingFace access](#add-huggingface-access)
- [Prepare data](#prepare-data)
  - [Audio files and transcripts](#audio-files-and-transcripts)
  - [Dataset card](#dataset-card)
- [Run the dataset builder and uploader script](#run-the-dataset-builder-and-uploader-script)
- [View dataset on HuggingFace](#view-dataset-on-huggingface)

## References

* https://huggingface.co/docs/datasets/en/audio_dataset
* https://huggingface.co/docs/hub/datasets-cards
* https://huggingface.co/docs/huggingface_hub/en/guides/cli#huggingface-cli-login


## Installation

The main script has dependencies.  Use a virtual environment such as `uv`.  In
this example I used Ubuntu 22.04 and Python `venv`.
```console
sudo apt install -y python3.10-venv

cd
python3 -m venv venv_test_hf_dataset
source ./venv_test_hf_dataset/bin/activate

cd test_hf_dataset

pip install --upgrade pip
pip install -r requirements.txt
```

## Add HuggingFace access

Authentication is needed with your HuggingFace account using HuggingFace's
command line interface tool.

First run the tool with the token to be saved as a git credential.
```console
cd

source ./venv_test_hf_dataset/bin/activate

# Adds store credential helper for adding the token as git credential
git config --global credential.helper store

huggingface-cli login
```
You'll need to get a token with the type `write` from
https://huggingface.co/settings/tokens and paste it to the command line, then
accept defaults.  Example result.
```
Add token as git credential? (Y/n) Y
Token is valid (permission: write).
The token `repo_test_hf_dataset` has been saved to /home/orangepi/.cache/huggingface/stored_tokens
Your token has been saved in your configured git credential helpers (store).
Your token has been saved to /home/orangepi/.cache/huggingface/token
Login successful.
The current active token is: `repo_test_hf_dataset`
```

## Prepare data

### Audio files and transcripts

Audio files (WAV or MP3) go in the `assets` folder.  Add these to the script
lists `file_names` and `text_names`.

### Dataset card

The HuggingFace dataset Dataset Card will use the markdown from
this repo's `DATASET_CARD.md`.  Edit the content as needed including YAML
metadata
[see HF url](https://huggingface.co/docs/hub/datasets-cards#dataset-card-metadata).

## Run the dataset builder and uploader script

```console
python3 main.py
```

## View dataset on HuggingFace

https://huggingface.co/datasets/guynich/test_hf_dataset
