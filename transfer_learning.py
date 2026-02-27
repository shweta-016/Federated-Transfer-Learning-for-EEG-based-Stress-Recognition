# transfer_learning.py
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from dataset_eeg import get_dataloader
from model_eegnet import EEGNet


def fine_tune_aggregated_model(
    agg_model_path: str,
    base_dir: str = "normalized_epochs",
    num_epochs: int = 5,
    lr: float = 1e-4,
    device: str = None,
    save_path: str = "final_finetuned.pth",
):
    """
    Fine-tunes the aggregated global model on server-side data
    (using subject_01 as example fine-tune dataset).
    """

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    agg_model_path = Path(agg_model_path)

    if not agg_model_path.exists():
        raise FileNotFoundError(f"Aggregated model file not found: {agg_model_path}")

    print(f"[Transfer] Loading aggregated model from: {agg_model_path}")

    # -------------------------------------------------
    # 1️⃣ Load one subject folder for fine-tuning
    # -------------------------------------------------
    subject_path = Path(base_dir) / "subject_01"

    if not subject_path.exists():
        raise FileNotFoundError(f"Subject folder not found: {subject_path}")

    val_loader, _ = get_dataloader(subject_path, batch_size=8, shuffle=True)

    # -------------------------------------------------
    # 2️⃣ Build model (we KNOW preprocessing config)
    # -------------------------------------------------
    num_channels = 32
    samples = 200
    num_classes = 2

    model = EEGNet(
        num_channels=num_channels,
        samples=samples,
        num_classes=num_classes,
    )

    state = torch.load(str(agg_model_path), map_location=device)

    # Load state dict safely
    if isinstance(state, dict) and all(isinstance(v, torch.Tensor) for v in state.values()):
        model.load_state_dict(state)
    elif isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    elif isinstance(state, dict) and "model_state" in state:
        model.load_state_dict(state["model_state"])
    else:
        model.load_state_dict(state)

    model.to(device)

    print("[Transfer] Aggregated model loaded successfully.")

    # -------------------------------------------------
    # 3️⃣ Freeze feature extractor (optional)
    # -------------------------------------------------
    for name, param in model.named_parameters():
        if "classify" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # If you want full fine-tuning instead:
    # for p in model.parameters():
    #     p.requires_grad = True

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )

    criterion = nn.CrossEntropyLoss()

    # -------------------------------------------------
    # 4️⃣ Fine-tune
    # -------------------------------------------------
    print("[Transfer] Starting fine-tuning...")

    for epoch in range(num_epochs):
        model.train()
        total = 0
        running_loss = 0.0

        for xb, yb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device).long()

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total += xb.size(0)
            running_loss += loss.item() * xb.size(0)

        epoch_loss = running_loss / total
        print(f"[Transfer] Epoch {epoch+1}/{num_epochs} Loss: {epoch_loss:.4f}")

    # -------------------------------------------------
    # 5️⃣ Save fine-tuned model
    # -------------------------------------------------
    torch.save(model.state_dict(), save_path)

    print(f"[Transfer] Fine-tuned model saved to: {save_path}")
    return save_path


if __name__ == "__main__":
    # Example standalone run
    agg_model_path = "server_saved_models/aggregated_round_3.pth"

    fine_tune_aggregated_model(
        agg_model_path=agg_model_path,
        num_epochs=3,
        lr=1e-4,
        save_path="final_finetuned.pth"
    )