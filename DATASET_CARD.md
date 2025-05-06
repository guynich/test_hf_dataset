---
language:
- en
pretty_name: test hf dataset
tags:
- speech
license: mit
task_categories:
- text-classification
---

# test_hf_dataset

This dataset was created to document how to create an audio dataset and upload
it to HuggingFace [see GitHub repo](https://github.com/guynich/test_hf_dataset).

Next step: add more documentation.
e.g.:
* contents of the dataset
* context for how the dataset should be used, e.g.: `datasets` package
* existing dataset cards, such as the ELI5 dataset card, show common conventions

# Example usage of dataset

Example of transcription.

First install extra dependencies, typically within virtual environment.
```
python3 -m pip install datasets torch transformers
```
Then save and run this Python script.  It runs transcription using the Moonshine
model by Useful Sensors [link](https://github.com/usefulsensors/moonshine).
```
"""Adapted from https://github.com/usefulsensors/moonshine#huggingface-transformers"""
from datasets import load_dataset
from transformers import AutoProcessor, MoonshineForConditionalGeneration

dataset = load_dataset("guynich/test_hf_dataset", split="test")
model = MoonshineForConditionalGeneration.from_pretrained(
    "UsefulSensors/moonshine-tiny"
)
processor = AutoProcessor.from_pretrained("UsefulSensors/moonshine-tiny")

for index in range(len(dataset)):
    audio_array = dataset[index]["audio"]["array"]
    sampling_rate = dataset[index]["audio"]["sampling_rate"]

    inputs = processor(audio_array, return_tensors="pt", sampling_rate=sampling_rate)

    generated_ids = model.generate(**inputs)

    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(transcription)
```
Example output.
```console
$ python3 main.py
The birch canoe slid on the smooth planks, glue the sheets to a dark blue background.
$
```
