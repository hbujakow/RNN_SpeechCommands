from copy import deepcopy
from pathlib import Path

import evaluate
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from transformers import TrainerCallback

CACHE_DIR = Path("../../cache_data")

accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)


class CustomCallback(TrainerCallback):

    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(
                eval_dataset=self._trainer.train_dataset, metric_key_prefix="train"
            )
            return control_copy


def train_epoch(device, model, criterion, optimizer, train_loader):
    model.train()
    running_loss = 0.0
    running_accuracy = 0.0

    for batch in train_loader:

        inputs, labels = batch["input_values"].to(device, non_blocking=True), batch[
            "label"
        ].to(device, non_blocking=True)
        optimizer.zero_grad()

        outputs = model(input_values=inputs).logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_accuracy += calculate_accuracy(outputs, labels) * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_accuracy = running_accuracy / len(train_loader.dataset)
    return epoch_loss, epoch_accuracy


def evaluate_model(
    device, model, criterion, test_loader, save_cm=False, cm_path=None, labels=None
):
    model.eval()
    running_loss = 0.0
    running_accuracy = 0.0
    predictions = []
    true_labels = []
    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch["input_values"].to(device, non_blocking=True), batch[
                "label"
            ].to(device, non_blocking=True)
            outputs = model(input_values=inputs).logits
            loss = criterion(outputs, labels)

            predictions.append(outputs)
            true_labels.append(labels)
            running_loss += loss.item() * inputs.size(0)
            running_accuracy += calculate_accuracy(outputs, labels) * inputs.size(0)

    try:
        if save_cm:
            predictions = torch.cat(predictions)
            true_labels = torch.cat(true_labels)
            predictions = torch.argmax(predictions, dim=1)
            true_labels = true_labels.cpu().numpy()
            predictions = predictions.cpu().numpy()
            save_confusion_matrix(true_labels, predictions, cm_path, labels)
    except:
        print("Error while plotting")

    test_loss = running_loss / len(test_loader.dataset)
    test_accuracy = running_accuracy / len(test_loader.dataset)
    return test_loss, test_accuracy


def calculate_accuracy(outputs, labels):
    outputs = torch.nn.functional.softmax(outputs, dim=-1)
    outputs = torch.argmax(outputs, dim=-1)
    corrects = torch.sum(outputs == labels.data).item()
    return corrects / labels.size(0)


def save_confusion_matrix(true_labels, predicted_labels, filename, labels):
    cm = confusion_matrix(true_labels, predicted_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, cmap="Blues")
    plt.xticks(rotation=45)
    plt.savefig(filename)
