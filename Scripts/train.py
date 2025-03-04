import torch
import torch.optim as optim
from torch.utils.data import  DataLoader
import torch.nn as nn
import pickle
import os
import torch.nn.functional as F
from EarlyStopping import EarlyStopping
from BathroomDataset import BathroomDataset
from BathroomPlacementModel import BathroomPlacementModel


def room_boundary_loss(pred_pos, room_width, room_length):
    batch_size = pred_pos.shape[0]
    room_width = room_width.expand(batch_size, 1)  # Expand to batch size
    room_length = room_length.expand(batch_size, 1)

    penalty = torch.where(
        (pred_pos[:, 0] < 0) | (pred_pos[:, 0] > room_width.squeeze(1)) |
        (pred_pos[:, 1] < 0) | (pred_pos[:, 1] > room_length.squeeze(1)),
        torch.tensor(10.0, device=pred_pos.device),
        torch.tensor(0.0, device=pred_pos.device)
    )
    return penalty.mean()

def fixture_specific_loss(outputs, targets, device, room_width,room_length):
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
        rot_loss = nn.CrossEntropyLoss()(pred_rot, target_rot)
        losses['rotation'] += rot_loss
        
        # Dimension verification
        # dims_penalty = F.mse_loss(pred_dims, fixture_dims[fixture].expand(batch_size, -1))
        # losses['dimension'] += dims_penalty
        boundary_loss = room_boundary_loss(pred_pos,room_width, room_length)
        
        # Combined loss with weights
        fixture_loss = pos_loss + 0.1 * rot_loss + boundary_loss
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
            
            loss, component_losses = fixture_specific_loss(outputs, targets, device,inputs[0][0],inputs[0][1])
            loss.backward()
            
            # Gradient clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
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
                
                loss, _ = fixture_specific_loss(outputs, targets, device,inputs[0][0],inputs[0][1])
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
    
    base_path = "/media/patrick/Patrick/Singularity_AI_Design_Tool/Data/augmented_bathroom_dataset_advanced"
    
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
    print(input_dim)
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
