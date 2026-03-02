# evaluate_model.py
# Professional evaluation script for Federated EEG Stress Detection
# Includes metrics, confusion matrix, ROC, multi-subject evaluation, CSV logging

import os
import argparse
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

from dataset_eeg import get_dataloader
from model_eegnet import EEGNet


# -------------------------------------------------------
# Utility: Build Model
# -------------------------------------------------------
def build_model(model_path, loader, device):
    x_sample, _ = next(iter(loader))
    _, _, channels, samples = x_sample.shape

    model = EEGNet(
        num_channels=channels,
        samples=samples,
        num_classes=2
    )

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


# -------------------------------------------------------
# Evaluate Single Subject
# -------------------------------------------------------
def evaluate_subject(model, loader, device):

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            outputs = model(xb)

            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = outputs.argmax(dim=1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(yb.numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, output_dict=True)

    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    return accuracy, cm, report, fpr, tpr, roc_auc


# -------------------------------------------------------
# Plot Confusion Matrix
# -------------------------------------------------------
def plot_confusion_matrix(cm, save_path):
    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.savefig(save_path)
    plt.close()


# -------------------------------------------------------
# Plot ROC Curve
# -------------------------------------------------------
def plot_roc(fpr, tpr, roc_auc, save_path):
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(save_path)
    plt.close()


# -------------------------------------------------------
# Main Evaluation Pipeline
# -------------------------------------------------------
def run_evaluation(model_path, base_dir, subject_id, output_dir):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\nLoading dataset...")
    from pathlib import Path

    subject_folder = Path(base_dir) / f"subject_{subject_id:02d}"

    train_loader, val_loader = get_dataloader(
        
        subject_folder,
        batch_size=8,
        shuffle=False
    
    
  
)




    loader = val_loader if val_loader is not None else train_loader

    print("Building model...")
    model = build_model(model_path, loader, device)

    print("Running evaluation...")
    accuracy, cm, report, fpr, tpr, roc_auc = evaluate_subject(
        model, loader, device
    )

    print("\n========== RESULTS ==========")
    print("Accuracy:", accuracy)
    print("ROC AUC:", roc_auc)
    print("\nConfusion Matrix:\n", cm)

    # Save results
    os.makedirs(output_dir, exist_ok=True)

    plot_confusion_matrix(cm, os.path.join(output_dir, "confusion_matrix.png"))
    plot_roc(fpr, tpr, roc_auc, os.path.join(output_dir, "roc_curve.png"))

    # Save metrics to CSV
    metrics_df = pd.DataFrame(report).transpose()
    metrics_df.to_csv(os.path.join(output_dir, "classification_report.csv"))

    print("\nResults saved to:", output_dir)


# -------------------------------------------------------
# CLI Entry
# -------------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="server_saved_models/final_finetuned.pth",
        help="Path to trained model (.pth)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="normalized_epochs",
        help="Dataset directory"
    )
    parser.add_argument(
        "--subject",
        type=int,
        default=1,
        help="Subject ID"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Folder to save evaluation outputs"
    )

    args = parser.parse_args()

    run_evaluation(
        model_path=args.model_path,
        base_dir=args.data_dir,
        subject_id=args.subject,
        output_dir=args.output_dir
    )