import json
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cuda:3'
subset_of_labels = ["up", "down", "left", "right", "on", "off", "yes", "no"]


def extract_fields(example):
    x = example["audio"]["array"]
    return {
        "label": example["label"],
        "array": np.pad(x, (0, 16000 - len(x)), constant_values=0),
    }


def preprocess(speech_data):
    speech_data = speech_data.filter(lambda x: len(x["audio"]["array"]) / 16_000 < 10)
    speech_data = speech_data.filter(
        lambda x: x["file"].split("/")[0] in subset_of_labels
    )

    train = speech_data["train"]
    validation = speech_data["validation"]
    test = speech_data["test"]

    train = train.map(extract_fields)
    validation = validation.map(extract_fields)
    test = test.map(extract_fields)

    train = train.map(remove_columns=["file", "audio", "speaker_id", "utterance_id"])
    validation = validation.map(
        remove_columns=["file", "audio", "speaker_id", "utterance_id"]
    )
    test = test.map(remove_columns=["file", "audio", "speaker_id", "utterance_id"])

    return train, validation, test

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))

        out = self.fc(out[:, -1, :])
        return out


def calculate_accuracy(preds, y):
    temp = torch.nn.functional.softmax(preds, dim=1)
    temp = torch.argmax(temp, dim=1)
    return (torch.sum(temp == y) / len(y)).item()


def main(config):
    BATCH_SIZE = config["batch_size"]
    NUM_EPOCHS = config["num_epochs"]
    SEED = config["seed"]
    NUM_LAYERS = config["num_layers"]

    speech_data = load_dataset("speech_commands", "v0.02")

    train, validation, test = preprocess(speech_data)

    train = train.shuffle(seed=SEED)
    validation = validation.shuffle(seed=SEED)
    test = test.shuffle(seed=SEED)

    torch.cuda.empty_cache()
    model = LSTMModel(16_000, 64, NUM_LAYERS, len(subset_of_labels))

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=NUM_EPOCHS, eta_min=0
    )
    best_val_loss = float("inf")

    results = pd.DataFrame(
        columns=["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"]
    )

    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(validation, batch_size=BATCH_SIZE)

    for epoch in tqdm(range(1, NUM_EPOCHS + 1)):
        model.train()
        train_loss = 0
        train_accuracy = 0
        for batch in train_loader:
            x = batch["array"]
            x = torch.stack(x).to(device)
            x = x.unsqueeze(1)
            x = x.permute(2, 1, 0)
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
        print(
            f"Epoch {epoch} train loss: {train_loss}, train accuracy: {train_accuracy}"
        )
        model.eval()
        with torch.no_grad():
            val_loss = 0
            val_accuracy = 0
            for batch in valid_loader:
                x = batch["array"]
                x = torch.stack(x).to(device)
                x = x.unsqueeze(1)
                x = x.permute(2, 1, 0)
                y = batch["label"].to(device)
                y_pred = model(x.float())
                val_loss += criterion(y_pred, y).item()
                val_accuracy += calculate_accuracy(y_pred, y)
            val_loss /= len(valid_loader)
            val_accuracy = val_accuracy / len(valid_loader)
            print(f"Epoch {epoch} val loss: {val_loss}, val accuracy: {val_accuracy}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), "best_model.pth")
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
    results.to_csv(f"results_{model._get_name()}_{NUM_LAYERS}.csv", index=False)


if __name__ == "__main__":
    with open("config.json") as f:
        config = json.load(f)
    main(config)
