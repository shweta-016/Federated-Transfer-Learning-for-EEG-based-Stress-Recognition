import os
import numpy as np
import pandas as pd

RAW_DIR = "SAM-40"
OUTPUT_DIR = "normalized_epochs"
WINDOW_SIZE = 200   # 200 time samples per epoch

os.makedirs(OUTPUT_DIR, exist_ok=True)

for file in os.listdir(RAW_DIR):

    if not file.endswith(".csv"):
        continue

    print(f"Processing {file}")

    # Extract subject number safely
    parts = file.split("_")

    subject_number = None
    for i, part in enumerate(parts):
        if part == "sub":
            subject_number = int(parts[i + 1])
            break

    if subject_number is None:
        continue

    subject_folder = os.path.join(
        OUTPUT_DIR, f"subject_{subject_number:02d}"
    )
    os.makedirs(subject_folder, exist_ok=True)

    # Label assignment
    if "Mirror_image" in file:
        label_name = "nonstress"
    else:
        label_name = "stress"

    df = pd.read_csv(os.path.join(RAW_DIR, file))

    eeg_data = df.values   # shape (32, 3200)

    # Normalize
    eeg_data = (eeg_data - np.mean(eeg_data)) / np.std(eeg_data)

    total_samples = eeg_data.shape[1]
    num_epochs = total_samples // WINDOW_SIZE

    for i in range(num_epochs):

        start = i * WINDOW_SIZE
        end = start + WINDOW_SIZE

        epoch = eeg_data[:, start:end]   # shape (32, 200)

        filename = f"{file.replace('.csv','')}_epoch_{i}_{label_name}.npy"

        np.save(os.path.join(subject_folder, filename), epoch)

print("SAM40 preprocessing completed successfully!")