import torch
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

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

def rotation_accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)


# Adjusted evaluation function to correctly extract target features
def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance on test data with correct target feature extraction.
    """
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test.values)
        y_true = torch.FloatTensor(y_test.values)

        y_pred = model(X_tensor).cpu().numpy()
        y_true = y_true.cpu().numpy()

        fixture_names = ['Toilet', 'Sink', 'Bathtub']
        rotations = np.array([0, 90, 180, 270])

        for i, fixture in enumerate(fixture_names):
            pos_start = i * 5
            rot_index = pos_start + 4

            # Extract actual and predicted positions
            true_pos = y_true[:, pos_start:pos_start + 2]
            pred_pos = y_pred[:, pos_start:pos_start + 2]

            # Compute MSE for positions
            mse = mean_squared_error(true_pos, pred_pos)
            print(f"{fixture} Position MSE: {mse:.4f}")

            # Extract actual and predicted rotations
            true_rot = y_true[:, rot_index]
            pred_rot_indices = np.argmin(np.abs(y_pred[:, rot_index][:, None] - rotations), axis=1)
            pred_rot = rotations[pred_rot_indices]

            # Compute accuracy for rotation
            rot_acc = rotation_accuracy_score(true_rot, pred_rot)
            print(f"{fixture} Rotation Accuracy: {rot_acc:.4f}")

            # Print actual vs. predicted
            print(f"True {fixture} positions:")
            print(true_pos[:5])
            print(f"Predicted {fixture} positions:")
            print(pred_pos[:5])
            print(f"True {fixture} rotations:")
            print(true_rot[:5])
            print(f"Predicted {fixture} rotations:")
            print(pred_rot[:5])
        # Visualize comparison of true vs predicted layouts
        visualize_comparison(X_test,y_true, y_pred)
        return y_pred


def normalize_rotation(rotation):
    """
    Normalize rotation to nearest valid angle (0, 90, 180, 270).
    File Location: /src/model/test.py
    """
    valid_rotations = [0, 90, 180, 270]
    # Find closest valid rotation
    return min(valid_rotations, key=lambda x: abs(float(rotation) - x))


def visualize_comparison(X_test, y_true, y_pred):
    """
    Visualize true vs predicted layout for the whole test batch.
    """
    # Ensure y_true and y_pred are NumPy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    num_samples = y_true.shape[0]
    for sample_idx in range(num_samples):
        # Extract room dimensions using NumPy indexing
        room_dims = {
            'width': X_test.iloc[sample_idx, 1],  # Correct way to get column 1 (Room Width)
            'length': X_test.iloc[sample_idx, 0]  # Correct way to get column 0 (Room Length)
        }

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        fixture_names = ['Toilet', 'Sink', 'Bathtub']

        # Extract positions and rotations correctly
        true_positions = [y_true[sample_idx, i*5:i*5+2] for i in range(3)]
        pred_positions = [y_pred[sample_idx, i*5:i*5+2] for i in range(3)]
        true_rotations = [y_true[sample_idx, i*5+4] for i in range(3)]
        pred_rotations = [y_pred[sample_idx, i*5+4] for i in range(3)]

        plot_layout(ax1, true_positions, true_rotations, room_dims, "True Layout")
        plot_layout(ax2, pred_positions, pred_rotations, room_dims, "Predicted Layout")

        plt.tight_layout()
        plt.show()


def plot_layout(ax, positions, rotations, room_dims, title):
    """
    Plot layout using rectangular patches instead of scatter plots.
    """
    ax.set_xlim(0, room_dims['width'])
    ax.set_ylim(0, room_dims['length'])
    ax.set_title(title)
    ax.set_xlabel("Width")
    ax.set_ylabel("Length")

    fixtures = ['Toilet', 'Sink', 'Bathtub']
    colors = ['red', 'blue', 'green']
    dimensions = [(19, 28), (30, 20), (30, 60)]  # (width, depth) for each fixture

    for i, (name, pos, rot, color, dim) in enumerate(zip(fixtures, positions, rotations, colors, dimensions)):
        width, height = dim if rot in [0, 180] else dim[::-1]  # Swap dimensions for 90° or 270° rotations
        rect = Rectangle(pos, width, height, angle=rot, color=color, alpha=0.5, label=name)
        ax.add_patch(rect)
        ax.text(pos[0] + width / 2, pos[1] + height / 2, f"{name}\n{int(rot)}°", ha='center', va='center', fontsize=10,
                bbox=dict(facecolor='white', alpha=0.7))

    ax.legend()


def main():
    """
    Main function for testing the trained model with debug prints.
    File Location: /src/model/test_model.py
    """
    try:
        print("Starting test process...")
        
        # Define base path
        base_path = "/media/patrick/Patrick/Singularity_AI_Design_Tool/Data/augmented_bathroom_dataset_advanced"
    
        print(f"Using base path: {base_path}")
        
        # Load test data
        print("\nAttempting to load test data...")
        test_x_path = os.path.join(base_path, 'X_test.csv')
        test_y_path = os.path.join(base_path, 'y_test.csv')
        
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
        
        # # Convert data to numeric
        # print("\nConverting data to numeric...")
        # for col in X_test.columns:
        #     X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0)
        # for col in y_test.columns:
        #     y_test[col] = pd.to_numeric(y_test[col], errors='coerce').fillna(0)
        
        # Convert to tensors
        print("\nConverting to tensors...")
        X_test_tensor = torch.FloatTensor(X_test.values.astype(np.float32))
        y_test_tensor = torch.FloatTensor(y_test.values.astype(np.float32))
        print(f"X_test_tensor shape: {X_test_tensor.shape}")
        print(f"y_test_tensor shape: {y_test_tensor.shape}")
        
        # Load the trained model
        print("\nLoading trained model...")
        model_path = "/media/patrick/Patrick/Singularity_AI_Design_Tool/Scripts/best_model.pth"
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
        evaluate_model(model,X_test,y_test)

    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()