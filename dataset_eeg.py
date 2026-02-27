import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from normalization import z_score_normalize

class EEGDataset(Dataset):
    """
    Dataset for loading preprocessed EEG epochs saved as .npy files.
    Expected structure:

    normalized_epochs/
        subject_01/
            file1_stress.npy
            file2_nonstress.npy
    """

    def __init__(self, subject_path):
        self.subject_path = Path(subject_path)

        if not self.subject_path.exists():
            raise FileNotFoundError(f"Subject path not found: {self.subject_path}")

        self.samples = []

        for file in os.listdir(self.subject_path):
            if file.endswith(".npy"):
                full_path = self.subject_path / file

                if "nonstress" in file.lower():
                    label = 0
                else:
                    label = 1

                self.samples.append((str(full_path), label))

        if len(self.samples) == 0:
            raise RuntimeError(f"No .npy files found in {self.subject_path}")

    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        data = np.load(file_path)
        # Z-score normalization
        data = z_score_normalize(data)
        
        data = torch.tensor(data, dtype=torch.float32)
        data = data.unsqueeze(0)
        label = torch.tensor(label, dtype=torch.long)
        return data, label

    
        
    

def get_dataloader(subject_path, batch_size=32, shuffle=True):
    dataset = EEGDataset(subject_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader, None