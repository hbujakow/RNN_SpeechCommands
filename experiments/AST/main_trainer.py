import json

import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import (AutoFeatureExtractor,
                          AutoModelForAudioClassification, Trainer,
                          TrainingArguments)
from utils import CustomCallback, compute_metrics


def main(config):
    HG_MODEL_PATH = config["hg_model_path"]
    BATCH_SIZE = config["batch_size"]
    NUM_EPOCHS = config["num_epochs"]
    SEED = config["seed"]

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

    speech_data = speech_data.filter(lambda x: len(x["audio"]["array"]) / 16_000 < 10)

    speech_data_processed = speech_data.map(
        preprocess_function,
        batched=True,
        remove_columns=["file", "audio", "speaker_id", "utterance_id", "is_unknown"],
    ).with_format("torch")

    labels = speech_data["train"].features["label"].names
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    num_labels = len(id2label)
    model = AutoModelForAudioClassification.from_pretrained(
        HG_MODEL_PATH,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )

    model_path = f"{HG_MODEL_PATH.replace('/', '_').replace('-', '_')}_seed_{SEED}"
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


if __name__ == "__main__":
    with open("config.json") as f:
        config = json.load(f)
    main(config)
