import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class BathroomDataset(Dataset):
    """
    Dataset class without normalization
    File Location: /src/model/train.py
    """

    def __init__(self, x_path, y_path):
        self.x_df = pd.read_csv(x_path)
        self.y_df = pd.read_csv(y_path)

        self.X = self.x_df.values.astype(np.float32)
        self.y = self.y_df.values.astype(np.float32)

        self.fixture_dims = {
            'toilet': {'width': 19.0, 'depth': 28.0},
            'sink': {'width': 30.0, 'depth': 20.0},
            'bathtub': {'width': 30.0, 'depth': 60.0}
        }

        print(f"Using fixed fixture dimensions: {self.fixture_dims}")
        print(f"X shape: {self.X.shape}, dtype: {self.X.dtype}")
        print(f"y shape: {self.y.shape}, dtype: {self.y.dtype}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.FloatTensor(self.y[idx])

    def get_room_dims(self, idx):
        return {
            'width': float(self.x_df['Room_Width'].iloc[idx]),
            'length': float(self.x_df['Room_Length'].iloc[idx])
        }