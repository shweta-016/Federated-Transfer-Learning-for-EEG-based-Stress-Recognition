# train_local.py
# Centralized training for comparison with Federated Learning

import torch
import torch.nn as nn
import torch.optim as optim
from dataset_eeg import get_dataloader
from model_eegnet import EEGNet


def train_local_model():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load one subject folder for centralized test
    subject_folder = "normalized_epochs/subject_01"

    train_loader, val_loader = get_dataloader(
        subject_folder,
        batch_size=8,
        shuffle=True
    )

    loader = val_loader if val_loader is not None else train_loader

    # Infer input shape
    x_sample, _ = next(iter(loader))
    _, _, channels, samples = x_sample.shape

    model = EEGNet(
        num_channels=channels,
        samples=samples,
        num_classes=2
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 20

    for epoch in range(epochs):
        model.train()
        total = 0
        correct = 0
        running_loss = 0

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device).long()

            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()

            preds = outputs.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)
            running_loss += loss.item() * xb.size(0)

        acc = correct / total
        print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/total:.4f} | Accuracy: {acc:.4f}")

    print("\nFinal Centralized Accuracy:", acc)


if __name__ == "__main__":
    train_local_model()