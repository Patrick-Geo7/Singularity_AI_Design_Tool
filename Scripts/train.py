import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
import pickle
import os
import torch.nn.functional as F
import numpy as np

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
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.001)
        )
        
        # Separate branches for position and rotation only
        self.position_branches = nn.ModuleDict({
            fixture: nn.Sequential(
                nn.Linear(256, 128), 
                nn.ReLU(),
                nn.Linear(128, 64), 
                nn.ReLU(),
                nn.Linear(64, 2)  # Predict x, y position
            ) for fixture in ['toilet', 'sink', 'bathtub']
        })

        self.rotation_branches = nn.ModuleDict({
            fixture: nn.Sequential(
                nn.Linear(256, 128), 
                nn.ReLU(),
                nn.Linear(128, 64), 
                nn.ReLU(),
                nn.Linear(64, 1),  # Predict rotation
                nn.Sigmoid()  # Scale output between 0 and 1
            ) for fixture in ['toilet', 'sink', 'bathtub']
        })
        self.spatial_features = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )


    def positional_encoding(self,x, max_len=5000):
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
        
        for fixture in fixtures:
            # Get position (x, y)
            pos = self.position_branches[fixture](shared)  # [batch_size, 2]
            
            # Get rotation and discretize it
            rot = self.rotation_branches[fixture](shared) * 360.0  # Scaling sigmoid output
            rot = self.discretize_rotation(rot)  # [batch_size, 1]
            
            # Get fixed dimensions and expand to batch size
            dims = self.fixture_dimensions[fixture].to(x.device)
            dims = dims.unsqueeze(0).expand(batch_size, -1)  # [batch_size, 2]
            
            # Combine position, dimensions, and rotation
            # Ensure all tensors have shape [batch_size, n]
            fixture_output = torch.cat([
                pos,                    # [batch_size, 2]
                dims,                   # [batch_size, 2]
                rot.view(batch_size, 1) # [batch_size, 1]
            ], dim=1)
            outputs.append(fixture_output)
        
        return torch.cat(outputs, dim=1)
    
    def discretize_rotation(self, rot):
        """
        Discretize rotation values to valid angles
        """
        rot = rot.squeeze(-1)  # Remove extra dimension if present
        valid_rots = torch.tensor([0., 90., 180., 270.], device=rot.device)
        rot_expanded = rot.unsqueeze(-1)
        diffs = torch.abs(rot_expanded - valid_rots)
        closest_idx = torch.argmin(diffs, dim=-1)
        return valid_rots[closest_idx]


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

class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop

def fixture_specific_loss(outputs, targets, device):
    """
    Enhanced loss function with dimension verification
    File Location: /src/model/train.py
    """
    batch_size = outputs.shape[0]
    total_loss = 0
    
    # Fixed dimensions for verification
    fixture_dims = {
        'toilet': torch.tensor([19.0, 28.0], device=device),
        'sink': torch.tensor([30.0, 20.0], device=device),
        'bathtub': torch.tensor([30.0, 60.0], device=device)
    }
    
    losses = {'position': 0, 'rotation': 0, 'dimension': 0}
    
    for idx, fixture in enumerate(['toilet', 'sink', 'bathtub']):
        start_idx = idx * 5
        
        # Extract components
        pred_pos = outputs[:, start_idx:start_idx + 2]
        pred_dims = outputs[:, start_idx + 2:start_idx + 4]
        pred_rot = outputs[:, start_idx + 4]
        
        target_pos = targets[:, start_idx:start_idx + 2]
        target_rot = targets[:, start_idx + 4]
        
        # Position loss
        pos_loss = F.mse_loss(pred_pos, target_pos)
        losses['position'] += pos_loss
        
        # Rotation loss
        rot_loss = F.mse_loss(pred_rot, target_rot)
        losses['rotation'] += rot_loss
        
        # Dimension verification
        dims_penalty = F.mse_loss(pred_dims, fixture_dims[fixture].expand(batch_size, -1))
        losses['dimension'] += dims_penalty
        
        # Combined loss with weights
        fixture_loss = pos_loss + 0.1 * rot_loss + 10.0 * dims_penalty
        total_loss += fixture_loss
    
    return total_loss / 3, losses

def verify_dimensions(outputs, fixture_dims, device):
    """
    Verify output dimensions match fixed values
    File Location: /src/model/train.py
    """
    dimension_errors = {}
    for idx, (fixture, dims) in enumerate(fixture_dims.items()):
        start_idx = idx * 5 + 2
        pred_dims = outputs[:, start_idx:start_idx + 2]
        dims_tensor = torch.tensor(dims, device=device).expand_as(pred_dims)
        diff = torch.abs(pred_dims - dims_tensor).mean()
        dimension_errors[fixture] = diff.item()
    return dimension_errors

def train_model(model, train_loader, val_loader, optimizer, num_epochs, device):
    """
    Enhanced training loop with detailed monitoring
    File Location: /src/model/train.py
    """
    best_val_loss = float('inf')
    early_stopping = EarlyStopping(patience=15, min_delta=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    train_losses = []
    val_losses = []
    dimension_errors = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        epoch_dim_errors = {'train': {}, 'val': {}}
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss, component_losses = fixture_specific_loss(outputs, targets, device)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            
            # Verify dimensions
            if batch_idx == 0:  # Check first batch of each epoch
                epoch_dim_errors['train'] = verify_dimensions(
                    outputs, model.fixture_dimensions, device
                )
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(val_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                
                loss, _ = fixture_specific_loss(outputs, targets, device)
                val_loss += loss.item()
                
                # Verify dimensions for first batch
                if batch_idx == 0:
                    epoch_dim_errors['val'] = verify_dimensions(
                        outputs, model.fixture_dimensions, device
                    )
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        dimension_errors.append(epoch_dim_errors)
        
        # Print progress
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {avg_train_loss:.4f}')
        print(f'Validation Loss: {avg_val_loss:.4f}')
        print('Dimension Errors:')
        print('Training:', epoch_dim_errors['train'])
        print('Validation:', epoch_dim_errors['val'])
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
                'dimension_errors': dimension_errors
            }, 'best_model.pth')
        
        # Early stopping check
        if early_stopping(avg_val_loss):
            print("Early stopping triggered")
            break
    
    return train_losses, val_losses, dimension_errors

def main():
    """
    Main execution function
    File Location: /src/model/train.py
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    base_path = "H:\\Shared drives\\AI Design Tool\\00-PG_folder\\03-Furniture AI Model\\Data\preprocessed"
    
    # Create datasets
    train_dataset = BathroomDataset(
        os.path.join(base_path, 'X_train.csv'),
        os.path.join(base_path, 'y_train.csv')
    )
    
    val_dataset = BathroomDataset(
        os.path.join(base_path, 'X_val.csv'),
        os.path.join(base_path, 'y_val.csv')
    )
    
    test_dataset = BathroomDataset(
        os.path.join(base_path, 'X_test.csv'),
        os.path.join(base_path, 'y_test.csv')
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Initialize model
    input_dim = train_dataset.X.shape[1]
    output_dim = train_dataset.y.shape[1]
    print(f"Model dimensions - Input: {input_dim}, Output: {output_dim}")
    
    model = BathroomPlacementModel(input_dim, output_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    
    # Train model
    train_losses, val_losses, dimension_errors = train_model(
        model, train_loader, val_loader, optimizer, 
        num_epochs=1000, device=device
    )
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'dimension_errors': dimension_errors
    }
    with open('training_history.pkl', 'wb') as f:
        pickle.dump(history, f)

if __name__ == "__main__":
    main()
