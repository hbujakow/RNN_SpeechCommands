import json
import os
import random
import warnings
from argparse import ArgumentParser
from math import ceil

import librosa
import numpy as np
import pandas as pd
import torch
from datasets import concatenate_datasets, load_dataset
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from models import GRUModel, LSTMModel, RNNModel

warnings.filterwarnings("ignore")

MODELS = {
    "rnn": RNNModel,
    "lstm": LSTMModel,
    "gru": GRUModel,
}

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_mfcc(data, sample_rate=16000, n_mfcc=12):
    # Extract MFCC features
    # https://librosa.github.io/librosa/generated/librosa.feature.mfcc.html
    mfcc = librosa.feature.mfcc(
        y=data,
        sr=sample_rate,
        n_mfcc=n_mfcc,  # How many mfcc features to use? 12 at most.
        # https://dsp.stackexchange.com/questions/28898/mfcc-significance-of-number-of-features
    )
    return mfcc


def extract_fields(example):
    x = example["audio"]["array"]
    return {
        "label": example["label"],
        "array": np.pad(x, (0, 16000 - len(x)), constant_values=0),
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

    train = train.map(
        extract_fields, remove_columns=["file", "audio", "speaker_id", "utterance_id"]
    )
    validation = validation.map(
        extract_fields, remove_columns=["file", "audio", "speaker_id", "utterance_id"]
    )
    test = test.map(
        extract_fields, remove_columns=["file", "audio", "speaker_id", "utterance_id"]
    )

    return (
        train.with_format("torch"),
        validation.with_format("torch"),
        test.with_format("torch"),
    )


def calculate_accuracy(preds, y):
    preds = torch.nn.functional.softmax(preds, dim=1)
    preds = torch.argmax(preds, dim=1)
    return (torch.sum(preds == y) / len(y)).item()


def main(config):
    BATCH_SIZE = config["batch_size"]
    NUM_EPOCHS = config["num_epochs"]
    SEED = config["seed"]

    speech_data = load_dataset(
        "speech_commands", "v0.02", cache_dir="only_selected/data_here"
    )

    train, validation, test = preprocess(speech_data)

    torch.cuda.empty_cache()

    hidden_sizes = [16, 32, 64, 128, 256]
    num_layers = list(range(1, 16))  # 1 to 15

    for NUM_LAYERS in num_layers:
        for HIDDEN_SIZE in hidden_sizes:
            torch.cuda.empty_cache()
            print(f"SEED: {SEED}, NUM_LAYERS: {NUM_LAYERS}, HIDDEN_SIZE: {HIDDEN_SIZE}")

            train = train.shuffle(seed=SEED)
            validation = validation.shuffle(seed=SEED)
            test = test.shuffle(seed=SEED)

            config["num_layers"] = NUM_LAYERS
            config["hidden_size"] = HIDDEN_SIZE

            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)

            model = MODELS[config["model"]](
                12, HIDDEN_SIZE, NUM_LAYERS, len(id2label)
            )  # 12 - corresponds to 12 features in MFCC

            model.to(device)

            model_path = (
                f"{model._get_name()}"
                + f"__NUM_LAY_{NUM_LAYERS}_HID_{HIDDEN_SIZE}_SEED_{SEED}"
            )

            if not os.path.exists(model_path):
                os.makedirs(model_path)

            with open(f"{model_path}/config.json", "w") as f:
                json.dump(config, f, indent=4)

            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, T_max=NUM_EPOCHS, eta_min=0
            )
            best_val_loss = float("inf")

            results = pd.DataFrame(
                columns=[
                    "epoch",
                    "train_loss",
                    "train_accuracy",
                    "val_loss",
                    "val_accuracy",
                ]
            )

            train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
            valid_loader = DataLoader(validation, batch_size=BATCH_SIZE)
            test_loader = DataLoader(test, batch_size=BATCH_SIZE)

            for epoch in tqdm(range(1, NUM_EPOCHS + 1)):
                model.train()
                train_loss = 0
                train_accuracy = 0
                for batch in train_loader:
                    x = batch["array"]
                    x = (
                        torch.Tensor(compute_mfcc(np.array(x), 16_000))
                        .permute(0, 2, 1)
                        .to(device)
                    )
                    y = batch["label"].to(device)
                    y_pred = model(x.float())
                    loss = criterion(y_pred, y)
                    train_loss += loss.item()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    train_accuracy += calculate_accuracy(y_pred, y)
                train_loss /= len(train_loader)
                train_accuracy = train_accuracy / len(train_loader)

                model.eval()
                with torch.no_grad():
                    val_loss = 0
                    val_accuracy = 0
                    for batch in valid_loader:
                        x = batch["array"]
                        x = (
                            torch.Tensor(compute_mfcc(np.array(x), 16_000))
                            .permute(0, 2, 1)
                            .to(device)
                        )
                        y = batch["label"].to(device)
                        y_pred = model(x.float())
                        val_loss += criterion(y_pred, y).item()
                        val_accuracy += calculate_accuracy(y_pred, y)
                    val_loss /= len(valid_loader)
                    val_accuracy = val_accuracy / len(valid_loader)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save(model.state_dict(), f"{model_path}/best_model.pth")
                    row = pd.DataFrame(
                        {
                            "epoch": [epoch],
                            "train_loss": [loss.item()],
                            "train_accuracy": [train_accuracy],
                            "val_loss": [val_loss],
                            "val_accuracy": [val_accuracy],
                        }
                    )
                    results = pd.merge(results, row, how="outer")
                print(
                    f"Epoch {epoch} train loss: {train_loss}, train accuracy: {train_accuracy} val loss: {val_loss}, val accuracy: {val_accuracy}"
                )
            results.to_csv(f"{model_path}/results.csv", index=False)

            model.eval()
            test_accuracy = 0.0
            predictions = []
            true_labels = []
            with torch.no_grad():
                for batch in test_loader:
                    x = batch["array"]
                    x = (
                        torch.Tensor(compute_mfcc(np.array(x), 16_000))
                        .permute(0, 2, 1)
                        .to(device)
                    )
                    y = batch["label"].to(device)
                    y_pred = model(x.float())
                    test_accuracy += calculate_accuracy(y_pred, y)
                    predictions.append(y_pred)
                    true_labels.append(y)
            test_accuracy = test_accuracy / len(test_loader)

            print(f"Test accuracy: {test_accuracy}")

            with open(f"{model_path}/test_results.json", "w") as f:
                json.dump({"test_accuracy": test_accuracy}, f, indent=4)


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--seed", type=int, default=1)
    args = argparser.parse_args()
    seed = args.seed
    with open("config.json") as f:
        config = json.load(f)
    config["seed"] = seed
    main(config)
