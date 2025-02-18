import torch
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from train import BathroomPlacementModel
import pickle
import os

def load_test_data(test_x_path: str, test_y_path: str):
    """
    Load test data from saved CSV files.
    """
    X_test = pd.read_csv(test_x_path)
    y_test = pd.read_csv(test_y_path)
    return X_test, y_test

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance on test data with raw values.
    """
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test.values)
        y_true = y_test.values
        
        y_pred = model(X_tensor).numpy()
        
        # Calculate MSE for each fixture using actual dimensions
        fixture_names = ['Toilet', 'Sink', 'Bathtub']
        for i, fixture in enumerate(fixture_names):
            mse = mean_squared_error(y_true[:, i*2:(i+1)*2], y_pred[:, i*2:(i+1)*2])
            print(f"{fixture} MSE: {mse:.4f}")
            # Print actual vs predicted positions
            print(f"True {fixture} positions:")
            print(y_true[:, i*2:(i+1)*2])
            print(f"Predicted {fixture} positions:")
            print(y_pred[:, i*2:(i+1)*2])
        
        return y_pred


def visualize_layout(y_true, y_pred, room_dims):
    """
    Visualize layout with actual dimensions.
    """
    print("True layout values (inches):")
    print(y_true)
    print("\nPredicted layout values (inches):")
    print(y_pred)

def normalize_rotation(rotation):
    """
    Normalize rotation to nearest valid angle (0, 90, 180, 270).
    File Location: /src/model/test.py
    """
    valid_rotations = [0, 90, 180, 270]
    # Find closest valid rotation
    return min(valid_rotations, key=lambda x: abs(float(rotation) - x))

def visualize_comparison(y_true, y_pred, room_dims, sample_idx=0):
    """
    Visualize true vs predicted layout with proper array handling.
    File Location: /src/model/test_model.py
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    print(f"\nSample {sample_idx + 1}:")
    print("True values:", y_true[sample_idx])
    print("Predicted values:", y_pred[sample_idx])
    
    # Extract positions and rotations correctly
    true_positions = [
        [y_true[sample_idx, 0], y_true[sample_idx, 1]],  # Toilet
        [y_true[sample_idx, 3], y_true[sample_idx, 4]],  # Sink
        [y_true[sample_idx, 6], y_true[sample_idx, 7]]   # Bathtub
    ]
    pred_rotations = [normalize_rotation(rot) for rot in 
                     [y_pred[sample_idx, 2], y_pred[sample_idx, 5], y_pred[sample_idx, 8]]]
    true_rotations = [normalize_rotation(rot) for rot in 
                     [y_true[sample_idx, 2], y_true[sample_idx, 5], y_true[sample_idx, 8]]]
    
    pred_positions = [
        [y_pred[sample_idx, 0], y_pred[sample_idx, 1]],  # Toilet
        [y_pred[sample_idx, 3], y_pred[sample_idx, 4]],  # Sink
        [y_pred[sample_idx, 6], y_pred[sample_idx, 7]]   # Bathtub
    ]
    
    # Plot layouts
    plot_layout(ax1, true_positions, true_rotations, room_dims, "True Layout")
    plot_layout(ax2, pred_positions, pred_rotations, room_dims, "Predicted Layout")
    
    plt.tight_layout()
    plt.show()

def plot_layout(ax, positions, rotations, room_dims, title):
    """
    Plot layout with enforced valid rotations.
    File Location: /src/model/test.py
    """
    room_width, room_length = room_dims['width'], room_dims['length']
    
    # Draw room boundary
    ax.add_patch(plt.Rectangle((0, 0), room_width, room_length, fill=False, color='black'))
    
    # Define fixtures with normalized rotations
    fixtures = [
        ('Toilet', positions[0], normalize_rotation(rotations[0]), (19, 28), 'red'),
        ('Sink', positions[1], normalize_rotation(rotations[1]), (30, 20), 'blue'),
        ('Bathtub', positions[2], normalize_rotation(rotations[2]), (30, 60), 'green')
    ]

    
    # Plot each fixture
    for name, pos, rotation, (width, depth), color in fixtures:
        # Create rectangle with explicit coordinates
        rect = plt.Rectangle(
            (float(pos[0]), float(pos[1])),  # Convert to float explicitly
            width,
            depth,
            angle=float(rotation),  # Convert rotation to float
            color=color,
            alpha=0.5,
            label=name
        )
        ax.add_patch(rect)
        
        # Add label
        ax.text(
            float(pos[0]) + width/2,
            float(pos[1]) + depth/2,
            f"{name}\n{width}\"x{depth}\"\nRot: {int(rotation)}°",
            ha='center',
            va='center'
        )
    
    # Set plot properties
    ax.set_xlim(-5, room_width + 5)
    ax.set_ylim(-5, room_length + 5)
    ax.grid(True)
    ax.set_title(title)
    ax.legend()

    
    def get_rotated_dims(width, depth, rotation):
        """Get width and depth based on rotation"""
        rot = rotation % 360
        if rot in [90, 270]:
            return depth, width
        return width, depth
    
    # Plot each fixture
    for name, pos, rotation, (width, depth), color in fixtures:
        # Get dimensions based on rotation
        actual_width, actual_depth = get_rotated_dims(width, depth, rotation)
        
        # Create rectangle
        rect = plt.Rectangle(
            (pos[0], pos[1]),
            actual_width,
            actual_depth,
            angle=rotation,
            color=color,
            alpha=0.5,
            label=name
        )
        ax.add_patch(rect)
        
        # Add label
        ax.text(
            pos[0] + actual_width/2,
            pos[1] + actual_depth/2,
            f"{name}\n{width}\"x{depth}\"\nRot: {int(rotation)}°",
            ha='center',
            va='center',
            bbox=dict(facecolor='white', alpha=0.7)
        )
    
    ax.set_title(title)
    ax.legend()

    # Print debug information
    print(f"\n{title} Debug Info:")
    for name, pos, rotation, (width, depth), _ in fixtures:
        print(f"{name}: Position ({pos[0]:.1f}, {pos[1]:.1f}), Rotation {rotation}°")


def main():
    """
    Main function for testing the trained model with debug prints.
    File Location: /src/model/test_model.py
    """
    try:
        print("Starting test process...")
        
        # Define base path
        base_path = "H:\\Shared drives\\AI Design Tool\\00-PG_folder\\03-Furniture AI Model"
    
        print(f"Using base path: {base_path}")
        
        # Load test data
        print("\nAttempting to load test data...")
        test_x_path = os.path.join(base_path+"\\Data\preprocessed", 'X_test.csv')
        test_y_path = os.path.join(base_path+"\\Data\preprocessed", 'y_test.csv')
        
        if not os.path.exists(test_x_path):
            print(f"Error: X_test.csv not found at {test_x_path}")
            return
        if not os.path.exists(test_y_path):
            print(f"Error: y_test.csv not found at {test_y_path}")
            return
            
        X_test = pd.read_csv(test_x_path)
        y_test = pd.read_csv(test_y_path)
        print(f"Loaded X_test shape: {X_test.shape}")
        print(f"Loaded y_test shape: {y_test.shape}")
        
        # Convert data to numeric
        print("\nConverting data to numeric...")
        for col in X_test.columns:
            X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0)
        for col in y_test.columns:
            y_test[col] = pd.to_numeric(y_test[col], errors='coerce').fillna(0)
        
        # Convert to tensors
        print("\nConverting to tensors...")
        X_test_tensor = torch.FloatTensor(X_test.values.astype(np.float32))
        y_test_tensor = torch.FloatTensor(y_test.values.astype(np.float32))
        print(f"X_test_tensor shape: {X_test_tensor.shape}")
        print(f"y_test_tensor shape: {y_test_tensor.shape}")
        
        # Load the trained model
        print("\nLoading trained model...")
        model_path = os.path.join(base_path+"\\Scripts\\", 'best_model.pth')
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            return
            
        checkpoint = torch.load(model_path)
        print("Model file loaded successfully")
        
        # Initialize model
        print("\nInitializing model...")
        input_dim = X_test.shape[1]
        output_dim = y_test.shape[1]
        print(f"Model dimensions - Input: {input_dim}, Output: {output_dim}")
        
        model = BathroomPlacementModel(input_dim, output_dim)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("Model initialized and set to eval mode")
        
        # Generate predictions
        print("\nGenerating predictions...")
        with torch.no_grad():
            predictions = model(X_test_tensor)
        print("Predictions generated successfully")
        
        # Room dimensions
        room_dims = {'width': 60, 'length': 96}
        
        # Visualize comparisons
        print("\nGenerating visualizations...")
        for i in range(min(5, len(y_test))):
            print(f"\nVisualizing sample {i+1}...")
            visualize_comparison(
                y_test.values,
                predictions.numpy(),
                room_dims,
                sample_idx=i
            )
            print(f"Sample {i+1} visualization complete")
        
        print("\nTest process completed successfully")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
