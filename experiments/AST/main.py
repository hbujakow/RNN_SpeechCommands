import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ASTForAudioClassification, AutoFeatureExtractor
from utils import CACHE_DIR, evaluate_model, train_epoch


def load_data(batch_size, seed, hg_model_path):
    feature_extractor = AutoFeatureExtractor.from_pretrained(hg_model_path)

    def preprocess_function(examples):
        audio_arrays = [x["array"] for x in examples["audio"]]
        inputs = feature_extractor(
            audio_arrays, sampling_rate=feature_extractor.sampling_rate, padding=True
        )
        return inputs

    train_data = load_dataset(
        "speech_commands", "v0.02", split="train", cache_dir=CACHE_DIR
    )
    validation_data = load_dataset(
        "speech_commands", "v0.02", split="validation", cache_dir=CACHE_DIR
    )
    test_data = load_dataset(
        "speech_commands", "v0.02", split="test", cache_dir=CACHE_DIR
    )

    class_labels = train_data.features["label"]
    subset_of_labels = ["up", "down", "left", "right", "on", "off", "yes", "no"]
    ids = [class_labels.str2int(label) for label in subset_of_labels]

    train_data = train_data.shuffle(seed=seed)
    validation_data = validation_data.shuffle(seed=seed)
    test_data = test_data.shuffle(seed=seed)

    train_data = train_data.map(
        preprocess_function,
        batched=True,
        remove_columns=["file", "audio", "speaker_id", "utterance_id"],
    ).with_format("torch")

    validation_data = validation_data.map(
        preprocess_function,
        batched=True,
        remove_columns=["file", "audio", "speaker_id", "utterance_id"],
    ).with_format("torch")

    test_data = test_data.map(
        preprocess_function,
        batched=True,
        remove_columns=["file", "audio", "speaker_id", "utterance_id"],
    ).with_format("torch")

    # train_data = train_data.filter(lambda x: x['label'] in ids)
    # validation_data = validation_data.filter(lambda x: x['label'] in ids)
    # test_data = test_data.filter(lambda x: x['label'] in ids)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validation_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    return train_loader, val_loader, test_loader, class_labels


def main(config):
    NUM_EPOCHS = config["num_epochs"]
    BATCH_SIZE = config["batch_size"]
    HG_MODEL_PATH = config["hg_model_path"]
    SEED = config["seed"]

    train_loader, val_loader, test_loader, class_labels = load_data(
        BATCH_SIZE, SEED, HG_MODEL_PATH
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{device = }")

    torch.cuda.empty_cache()
    model = ASTForAudioClassification.from_pretrained(HG_MODEL_PATH)
    model.classifier.dense = torch.nn.Linear(
        in_features=768, out_features=36, bias=True
    )  # changing number of classes from 35 to 36
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=NUM_EPOCHS, eta_min=0
    )

    best_val_loss = float("inf")
    CHECKPOINTS_PATH = Path("checkpoints")
    model_name = HG_MODEL_PATH.replace("/", "_").replace("-", "_")
    current_experiment_folder_run = (
        f"{model_name}_seed_{SEED}_date_{datetime.now().strftime('%Y_%m_%d_%H')}"
    )

    columns = ["Train Loss", "Train Accuracy", "Val Loss", "Val Accuracy"]
    df = pd.DataFrame(columns=columns)

    if not os.path.exists(CHECKPOINTS_PATH / current_experiment_folder_run):
        os.makedirs(CHECKPOINTS_PATH / current_experiment_folder_run)

    for epoch in tqdm(range(1, NUM_EPOCHS + 1)):
        train_loss, train_accuracy = train_epoch(
            device, model, criterion, optimizer, train_loader
        )
        val_loss, val_accuracy = evaluate_model(device, model, criterion, val_loader)

        # Checkpoint saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_name = (
                f"{model_name}_epoch_{epoch}_seed_{SEED}_val_loss_{val_loss:.4f}.pth"
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                },
                os.path.join(
                    CHECKPOINTS_PATH / current_experiment_folder_run, checkpoint_name
                ),
            )

        scheduler.step()

        df.loc[epoch] = [train_loss, train_accuracy, val_loss, val_accuracy]
        print(
            f"Epoch [{epoch}/{NUM_EPOCHS}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}"
        )

    test_loss, test_accuracy = evaluate_model(
        device,
        model,
        criterion,
        test_loader,
        save_cm=True,
        cm_path=f"{CHECKPOINTS_PATH / current_experiment_folder_run}/confusion_matrix.png",
        labels=class_labels,
    )
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    with open("config.json") as f:
        config = json.load(f)
    main(config)
