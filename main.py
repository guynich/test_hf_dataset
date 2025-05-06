"""Creates and uploads an audio dataset to HuggingFace hub."""

import os
from pathlib import Path
from typing import List

import datasets as ds
import soundfile as sf
from huggingface_hub import HfApi

# Constants
HF_PATH = "guynich/test_hf_dataset"  # Replace with your HuggingFace username and dataset name.
SPLIT_NAME = "test"
SAMPLE_RATE = 16000


def get_audio_duration(file_path: Path) -> float:
    """Calculate duration of audio file in seconds."""
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    audio, fs = sf.read(str(file_path))
    return len(audio) / fs


def create_audio_dataset(
    file_names: List[str], text_names: List[str], asset_dir: Path = Path("assets")
) -> ds.Dataset:
    """Create a HuggingFace dataset from audio files and transcripts.

    Args:
        file_names: List of audio file names
        text_names: List of corresponding transcript text file names
        asset_dir: Directory containing audio and text files

    Returns:
        HuggingFace dataset with audio and text
    """
    if len(file_names) != len(text_names):
        raise ValueError("Number of audio files must match number of text files.")

    file_paths = [asset_dir / name for name in file_names]
    time_secs = [get_audio_duration(path) for path in file_paths]

    # Read text files
    texts = []
    for text_name in text_names:
        with open(os.path.join(asset_dir, text_name), "r", encoding="utf-8") as f:
            texts.append(f.read().strip())

    return ds.Dataset.from_dict(
        {"audio": [str(p) for p in file_paths], "text": texts, "time_secs": time_secs},
        split=SPLIT_NAME,
    ).cast_column("audio", ds.Audio(sampling_rate=SAMPLE_RATE))


def main():
    # Input data lists
    file_names = ["harvard_sentences.wav"]  # Audio file(s)
    text_names = ["harvard_sentences.txt"]  # Corresponding transcript(s)

    # Create and process dataset
    audio_dataset = create_audio_dataset(file_names, text_names)

    # Optional: inspect an example for debugging
    # print(f"Sample audio data:\n{audio_dataset[0]['audio']}\n")

    # Push to hub
    audio_dataset.push_to_hub(
        HF_PATH,
        private=False,
    )

    # Verify by loading back
    loaded_ds = ds.load_dataset(HF_PATH)
    print(f"\nLoaded dataset:\n{loaded_ds}\n")
    print(f"First sample:\n{loaded_ds[SPLIT_NAME][0]}")

    # Upload Dataset Card
    api = HfApi()
    api.upload_file(
        path_or_fileobj="DATASET_CARD.md",
        path_in_repo="README.md",
        repo_id=HF_PATH,
        repo_type="dataset",
    )


if __name__ == "__main__":
    main()
