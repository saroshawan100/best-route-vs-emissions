# This script will handle loading those two year .npz files we just generated.
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class TrafficDataset(Dataset):
    def __init__(self, npz_file_path):
        """
        Loads the pre-processed .npz dataset.
        x shape: (num_samples, seq_len, num_nodes, features)
        y shape: (num_samples, horizon, num_nodes, features)
        """
        print(f"Loading dataset from {npz_file_path}...")
        data = np.load(npz_file_path)
        
        # Load data and convert to float32 for PyTorch training
        self.x = torch.tensor(data['x'], dtype=torch.float32)
        self.y = torch.tensor(data['y'], dtype=torch.float32)
        
        self.x = torch.nan_to_num(self.x, nan=0.0, posinf=0.0, neginf=0.0)
        self.y = torch.nan_to_num(self.y, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Keeps 20,000 time steps and first 500 highway nodes
        self.x = self.x[:20000, :, :500, :]
        self.y = self.y[:20000, :, :500, 0]
        # We only want to predict the first feature (speed)
        # Assuming shape is (N, T, V, F), we extract F=0 for labels
        #self.y = self.y[..., 0] 
        
        print(f"Dataset loaded. X: {self.x.shape}, Y: {self.y.shape}")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def get_dataloaders(data_dir, batch_size=32):
    """
    Creates PyTorch DataLoaders for train, val, and test splits.
    """
    train_dataset = TrafficDataset(f"{data_dir}/train.npz")
    val_dataset = TrafficDataset(f"{data_dir}/val.npz")
    test_dataset = TrafficDataset(f"{data_dir}/test.npz")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader