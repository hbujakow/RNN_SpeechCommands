import json
import random
import warnings
from math import ceil

import numpy as np
import pandas as pd
import torch
from datasets import DatasetDict, concatenate_datasets, load_dataset
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    Trainer,
    TrainingArguments,
)
from utils import CustomCallback, compute_metrics

warnings.filterwarnings("ignore")

normal_classes = {"yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"}
to_be_subsampled = {
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "bed",
    "bird",
    "cat",
    "dog",
    "happy",
    "house",
    "marvin",
    "sheila",
    "tree",
    "wow",
    "backward",
    "forward",
    "follow",
    "learn",
    "visual",
}


def swap_to_unknown(examples):
    examples["label"] = [10] * len(examples["label"])
    return examples


def swap_to_silence(examples):
    examples["label"] = [11] * len(examples["label"])
    return examples


def cut_audio(example):

    if len(example["audio"]["array"]) > 16000:
        example["audio"]["array"] = example["audio"]["array"][:16000]

    return example


def swap_labels_for_data_split(data):
    id2label_org = data.features["label"].int2str
    normal = data.filter(lambda x: id2label_org(x["label"]) in normal_classes)

    class_size = ceil(len(normal) / len(normal_classes))
    to_subsample = data.filter(lambda x: id2label_org(x["label"]) in to_be_subsampled)
    silence = data.filter(lambda x: id2label_org(x["label"]) == "_silence_")

    silence = silence.map(cut_audio)
    to_subsample = to_subsample.map(swap_to_unknown, batched=True)

    if len(to_subsample) > class_size:
        to_subsample = to_subsample.shuffle(seed=42).select(range(class_size))

    to_subsample = to_subsample.map(swap_to_unknown, batched=True)
    silence = silence.map(swap_to_silence, batched=True)
    results = concatenate_datasets([normal, to_subsample, silence])
    return results


def preprocess(speech_data):
    train = speech_data["train"]
    validation = speech_data["validation"]
    test = speech_data["test"]

    train = swap_labels_for_data_split(train)
    validation = swap_labels_for_data_split(validation)
    test = swap_labels_for_data_split(test)

    return DatasetDict({"train": train, "validation": validation, "test": test})


def main(config):
    HG_MODEL_PATH = config["hg_model_path"]
    BATCH_SIZE = config["batch_size"]
    NUM_EPOCHS = config["num_epochs"]
    SEED = config["seed"]

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    feature_extractor = AutoFeatureExtractor.from_pretrained(HG_MODEL_PATH)

    def preprocess_function(examples):
        audio_arrays = [x["array"] for x in examples["audio"]]
        inputs = feature_extractor(
            audio_arrays, sampling_rate=feature_extractor.sampling_rate, padding=True
        )
        return inputs

    speech_data = load_dataset(
        "speech_commands", "v0.02", cache_dir="../../cache_data_whole"
    )

    speech_data = preprocess(speech_data)

    speech_data_processed = speech_data.map(
        preprocess_function,
        batched=True,
        remove_columns=["file", "audio", "speaker_id", "utterance_id", "is_unknown"],
    ).with_format("torch")

    id2label = {
        "0": "yes",
        "1": "no",
        "2": "up",
        "3": "down",
        "4": "left",
        "5": "right",
        "6": "on",
        "7": "off",
        "8": "stop",
        "9": "go",
        "10": "_unknown_",
        "11": "_silence_",
    }
    label2id = {v: k for k, v in id2label.items()}

    if config["pretrained"]:
        num_labels = len(id2label)
        model = AutoModelForAudioClassification.from_pretrained(
            HG_MODEL_PATH,
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label,
            ignore_mismatched_sizes=True,
        )
    else:
        model_config = AutoConfig.from_pretrained(HG_MODEL_PATH)
        model_config.id2label = id2label
        model_config.label2id = label2id
        model = AutoModelForAudioClassification.from_config(model_config)

    pretrained = "pretrained" if config["pretrained"] else "not_pretrained"
    model_path = (
        f"{pretrained}/{HG_MODEL_PATH.replace('/', '_').replace('-', '_')}_seed_{SEED}"
    )
    training_args = TrainingArguments(
        output_dir=f"checkpoints/{model_path}",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model.to(device),
        args=training_args,
        train_dataset=speech_data_processed["train"],
        eval_dataset=speech_data_processed["validation"],
        tokenizer=feature_extractor,
        compute_metrics=compute_metrics,
    )
    trainer.add_callback(CustomCallback(trainer))

    trainer.train()

    train_losses = [
        {
            k: v
            for k, v in dictionary.items()
            if k in ("train_loss", "train_accuracy", "epoch")
        }
        for dictionary in trainer.state.log_history
        if "train_accuracy" in dictionary
    ]
    train_losses = pd.DataFrame(train_losses)
    train_losses.epoch = train_losses.epoch.astype(int)

    val_losses = [
        {
            k: v
            for k, v in dictionary.items()
            if k in ("eval_loss", "eval_accuracy", "epoch")
        }
        for dictionary in trainer.state.log_history
        if "eval_accuracy" in dictionary
    ]
    val_losses = pd.DataFrame(val_losses)
    val_losses.epoch = val_losses.epoch.astype(int)

    train_losses.to_csv(f"checkpoints/{model_path}/train_losses.csv", index=False)
    val_losses.to_csv(f"checkpoints/{model_path}/val_losses.csv", index=False)

    output = trainer.predict(speech_data_processed["test"])

    with open(f"checkpoints/{model_path}/test_metrics.json", "w") as f:
        json.dump(output.metrics, f, indent=4)

    with open(f"checkpoints/{model_path}/config.json", "w") as f:
        json.dump(config, f, indent=4)


if __name__ == "__main__":
    with open("config.json") as f:
        config = json.load(f)
    main(config)
