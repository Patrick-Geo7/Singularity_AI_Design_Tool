import torch
import torch.nn as nn
import numpy as np
from sympy import rotations


class BathroomPlacementModel(nn.Module):
    """
    Fixed model with correct tensor dimensions
    File Location: /src/model/train.py
    """

    def __init__(self, input_dim, output_dim):
        super(BathroomPlacementModel, self).__init__()

        # Fixed dimensions for fixtures
        self.fixture_dimensions = {
            'toilet': torch.tensor([19.0, 28.0]),  # width, depth
            'sink': torch.tensor([30.0, 20.0]),
            'bathtub': torch.tensor([30.0, 60.0])
        }

        self.shared_features = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.1)
        )

        # Separate branches for position and rotation only
        self.position_branches = nn.ModuleDict({
            fixture: nn.Sequential(
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 2)
            ) for fixture in ['toilet', 'sink', 'bathtub']
        })

        self.rotation_branches = nn.ModuleDict({
            fixture: nn.Sequential(
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 8),
                nn.ReLU(),
                nn.Linear(8, 4),  # 4 possible rotation angles
                nn.Softmax(dim=1)  # Outputs probabilities for {0°, 90°, 180°, 270°}
            ) for fixture in ['toilet', 'sink', 'bathtub']
        })

        self.spatial_features = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

    def positional_encoding(self, x, max_len=5000):
        """
        Applies sinusoidal positional encoding to enhance spatial awareness
        """
        pe = torch.zeros(max_len, x.shape[1]).to(x.device)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1).to(x.device)
        div_term = torch.exp(torch.arange(0, x.shape[1], 2).float() * (-np.log(10000.0) / x.shape[1])).to(x.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return x + pe[:x.shape[0], :]

    def forward(self, x):
        shared = self.shared_features(x)
        shared = self.positional_encoding(shared)

        batch_size = x.shape[0]

        outputs = []
        fixtures = ['toilet', 'sink', 'bathtub']
        rotations = torch.tensor([0, 90, 180, 270], device=x.device)
        for fixture in fixtures:
            # Get position (x, y)
            pos = self.position_branches[fixture](shared)  # [batch_size, 2]

            # Get rotation and discretize it
            rot_probs = self.rotation_branches[fixture](shared)  # Scaling sigmoid output
            rot_idices = torch.argmax(rot_probs, dim=1) # Get the index of the highest probability
            rot = rotations[rot_idices].unsqueeze(1)

            # Get fixed dimensions and expand to batch size
            dims = self.fixture_dimensions[fixture].to(x.device)
            dims = dims.unsqueeze(0).expand(batch_size, -1)  # [batch_size, 2]

            # Combine position, dimensions, and rotation
            # Ensure all tensors have shape [batch_size, n]
            fixture_output = torch.cat([pos, dims, rot], dim=1)  # [batch_size, 5]
            outputs.append(fixture_output)

        return torch.cat(outputs, dim=1)

    # def discretize_rotation(self, rot):
    #     """
    #     Discretize rotation values to valid angles
    #     """
    #     rot = rot.squeeze(-1)  # Remove extra dimension if present
    #     valid_rots = torch.tensor([0., 90., 180., 270.], device=rot.device)
    #     rot_expanded = rot.unsqueeze(-1)
    #     diffs = torch.abs(rot_expanded - valid_rots)
    #     closest_idx = torch.argmin(diffs, dim=-1)
    #     return valid_rots[closest_idx]