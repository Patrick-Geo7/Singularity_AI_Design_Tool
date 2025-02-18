import pandas as pd
import matplotlib.pyplot as plt              # NEW import for visualization
from matplotlib.patches import Rectangle     # NEW import for drawing rectangles
from sklearn.model_selection import train_test_split
import os                                    # NEW import for file saving

def visualize_layouts(df: pd.DataFrame, output_dir="visualizations"):
    """
    Visualize layouts with fixtures respecting their rotation angles.
    Code Location: /src/data/preprocess.py
    
    Rotation handling:
    0° or 180°: width parallel to room width, depth parallel to room length
    90° or 270°: width parallel to room length, depth parallel to room width
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for index, row in df.iterrows():
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_title(f"Layout {row.get('Room_ID', index)}")
        
        # Draw room boundary
        room_width = row['Room_Width']
        room_length = row['Room_Length']
        room = Rectangle((0, 0), room_width, room_length, 
                        edgecolor='black', facecolor='none', lw=2)
        ax.add_patch(room)
        
        # Draw origin marker
        ax.plot(0, 0, 'ko', markersize=10)
        ax.text(-0.5, -0.5, '(0,0)', fontsize=10)
        
        # Draw door width
        door_x = row['Door_X_Position']
        door_y = row['Door_Y_Position']
        door_width = row['Door_Width']
        door = Rectangle((door_x, door_y), door_width, 0.2,
                        edgecolor='blue', facecolor='blue', alpha=0.5)
        ax.add_patch(door)
        ax.text(door_x, door_y + 0.3, 
                f"Door\nX: {door_x:.2f}, Y: {door_y:.2f}\nWidth: {door_width:.2f}", 
                color='blue', fontsize=8)
        
        # Draw fixtures as rectangles with dimensions
        fixtures = [
            {
                'name': 'Toilet',
                'color': 'red',
                'x': row['Toilet_X_Position'],
                'y': row['Toilet_Y_Position'],
                'width': row['Toilet_Width'],
                'depth': row['Toilet_Depth'],
                'rotation': row['Toilet_Rotation'],
                'has_fixture': row['Has_Toilet']
            },
            {
                'name': 'Sink',
                'color': 'green',
                'x': row['Sink_X_Position'],
                'y': row['Sink_Y_Position'],
                'width': row['Sink_Width'],
                'depth': row['Sink_Depth'],
                'rotation': row['Sink_Rotation'],
                'has_fixture': row['Has_Sink']
            },
            {
                'name': 'Bathtub',
                'color': 'purple',
                'x': row['Bathtub_X_Position'],
                'y': row['Bathtub_Y_Position'],
                'width': row['Bathtub_Width'],
                'depth': row['Bathtub_Depth'],
                'rotation': row['Bathtub_Rotation'],
                'has_fixture': row['Has_Bathtub']
            }
        ]
        
        for fixture in fixtures:
            if fixture['has_fixture']:
                # Determine dimensions based on rotation
                if fixture['rotation'] in [0, 180]:
                    rect_width = fixture['width']
                    rect_depth = fixture['depth']
                else:  # 90 or 270 degrees
                    rect_width = fixture['depth']
                    rect_depth = fixture['width']
                
                # Draw fixture as rectangle
                fix_rect = Rectangle(
                    (fixture['x'], fixture['y']),
                    rect_width,
                    rect_depth,
                    edgecolor=fixture['color'],
                    facecolor=fixture['color'],
                    alpha=0.3
                )
                ax.add_patch(fix_rect)
                
                # Add fixture information
                ax.text(fixture['x'], fixture['y'] + rect_depth + 0.2,
                       f"{fixture['name']}\nX: {fixture['x']:.2f}, Y: {fixture['y']:.2f}\n"
                       f"Width: {fixture['width']:.2f}, Depth: {fixture['depth']:.2f}\n"
                       f"Rotation: {fixture['rotation']}°",
                       color=fixture['color'], fontsize=8)
                
                # Draw dimension lines
                # Width dimension
                ax.arrow(fixture['x'], fixture['y'] - 1, rect_width, 0,
                        head_width=0.3, head_length=0.3, fc=fixture['color'], ec=fixture['color'])
                ax.arrow(fixture['x'] + rect_width, fixture['y'] - 1, -rect_width, 0,
                        head_width=0.3, head_length=0.3, fc=fixture['color'], ec=fixture['color'])
                
                # Depth dimension
                ax.arrow(fixture['x'] - 1, fixture['y'], 0, rect_depth,
                        head_width=0.3, head_length=0.3, fc=fixture['color'], ec=fixture['color'])
                ax.arrow(fixture['x'] - 1, fixture['y'] + rect_depth, 0, -rect_depth,
                        head_width=0.3, head_length=0.3, fc=fixture['color'], ec=fixture['color'])
                
                # Draw rotation indicator (arrow showing direction)
                center_x = fixture['x'] + rect_width/2
                center_y = fixture['y'] + rect_depth/2
                rotation_length = min(rect_width, rect_depth) * 0.3
                
                # Calculate arrow direction based on rotation
                if fixture['rotation'] == 0:    # Pointing up
                    dx, dy = 0, rotation_length
                elif fixture['rotation'] == 90:  # Pointing right
                    dx, dy = rotation_length, 0
                elif fixture['rotation'] == 180: # Pointing down
                    dx, dy = 0, -rotation_length
                else:                           # Pointing left
                    dx, dy = -rotation_length, 0
                
                ax.arrow(center_x, center_y, dx, dy,
                        head_width=0.3, head_length=0.3,
                        fc=fixture['color'], ec=fixture['color'])
        
        # Draw coordinate system
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # Set axis limits with padding
        padding = max(room_width, room_length) * 0.2
        ax.set_xlim(-padding, room_width + padding)
        ax.set_ylim(-padding, room_length + padding)
        ax.set_aspect('equal', adjustable='box')
        
        # Label axes
        ax.set_xlabel("X (Width) - Measured from left")
        ax.set_ylabel("Y (Length) - Measured from bottom")
        
        # Add verification text
        verification_text = (
            f"Room Dimensions: {room_width:.2f} x {room_length:.2f}\n"
            f"Origin: Bottom Left (0,0)\n"
            f"All measurements from bottom left"
        )
        ax.text(room_width/2, -padding/2, verification_text,
                horizontalalignment='center', verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
        # Save the figure
        output_path = os.path.join(output_dir, f"layout_{row.get('Room_ID', index)}.png")
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        print(f"Saved visualization for layout {row.get('Room_ID', index)} to {output_path}")

def load_and_split_data(filepath: str, output_dir: str = "processed_data"):
    """
    Load and process bathroom layout data with proper file handling.
    File Location: /src/data/preprocess.py
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        df = pd.read_csv(filepath)
        print("Data loaded successfully, shape:", df.shape)
        
        # NEW: Visualize all layouts in the dataset (without denormalizing data)
        visualize_layouts(df)
        
        # Enhanced rotation verification and correction
        def verify_and_correct_rotation(value):
            """Helper function to ensure rotation values are valid"""
            valid_rotations = [0, 90, 180, 270]
            if pd.isna(value):
                return 0  # Default to 0 for missing values
            value = float(value)
            # Normalize rotation to 0-360 range
            value = value % 360
            # Find closest valid rotation
            return min(valid_rotations, key=lambda x: abs(x - value))

        # Apply rotation verification to all fixtures
        for fixture in ['Toilet', 'Sink', 'Bathtub']:
            rot_col = f'{fixture}_Rotation'
            if rot_col in df.columns:
                # Store original values for comparison
                original_values = df[rot_col].copy()
                # Apply correction
                df[rot_col] = df[rot_col].apply(verify_and_correct_rotation)
                # Report any corrections made
                changed_values = df[rot_col] != original_values
                if changed_values.any():
                    print(f"\nCorrected {fixture} rotations:")
                    for idx in df[changed_values].index:
                        print(f"Row {idx}: {original_values[idx]} -> {df[rot_col][idx]}")

        # Your existing input and target features
        input_features = [
            # Room properties
            'Room_Length', 
            'Room_Width',
            # Door information
            'Door_X_Position',
            'Door_Y_Position',
            'Door_Width',
            # Fixture presence flags
            'Has_Toilet',
            'Has_Sink',
            'Has_Bathtub'
        ]

        target_features = [
            # Toilet position and rotation
            'Toilet_X_Position',
            'Toilet_Y_Position',
            'Toilet_Width',
            'Toilet_Depth',
            'Toilet_Rotation',
            # Sink position and rotation
            'Sink_X_Position',
            'Sink_Y_Position',
            'Sink_Width',
            'Sink_Depth',
            'Sink_Rotation',
            # Bathtub position and rotation
            'Bathtub_X_Position',
            'Bathtub_Y_Position',
            'Bathtub_Width',
            'Bathtub_Depth',
            'Bathtub_Rotation'
        ]

        # Verify rotation values with detailed reporting
        rotation_summary = {}
        for fixture in ['Toilet', 'Sink', 'Bathtub']:
            rot_col = f'{fixture}_Rotation'
            if rot_col in df.columns:
                rotation_summary[fixture] = {
                    'values': df[rot_col].value_counts().to_dict(),
                    'min': df[rot_col].min(),
                    'max': df[rot_col].max()
                }
                print(f"\n{fixture} Rotation Summary:")
                print(f"Distribution: {rotation_summary[fixture]['values']}")
                print(f"Range: {rotation_summary[fixture]['min']}° to {rotation_summary[fixture]['max']}°")

        # Create feature matrices
        X = df[input_features]
        y = df[target_features]

        print("\nInput Features:")
        for col in input_features:
            print(f"- {col}")
        
        print("\nTarget Features:")
        for col in target_features:
            print(f"- {col}")

        # Split data with stratification by rotation if possible
        try:
            # Combine all rotations for stratification
            combined_rot = df['Toilet_Rotation'].astype(str) + '_' + \
                         df['Sink_Rotation'].astype(str) + '_' + \
                         df['Bathtub_Rotation'].astype(str)
            
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=combined_rot
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.5, random_state=42,
                stratify=combined_rot[X_temp.index]
            )
        except:
            # Fallback to regular split if stratification fails
            print("\nFalling back to regular split (stratification failed)")
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.5, random_state=42
            )

        # Save splits with enhanced verification
        splits = {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }

        for name, data in splits.items():
            try:
                output_path = os.path.join(output_dir, f"{name}.csv")
                data.to_csv(output_path, index=False)
                print(f"\nSaved {output_path}")
                print(f"Shape: {data.shape}")
                if name.startswith('y_'):
                    print("Rotation distribution:")
                    for fixture in ['Toilet', 'Sink', 'Bathtub']:
                        rot_col = f'{fixture}_Rotation'
                        if rot_col in data.columns:
                            dist = data[rot_col].value_counts().to_dict()
                            print(f"{fixture}: {dist}")
            except PermissionError:
                print(f"Permission denied when saving {name}.csv. Please check folder permissions.")
                print(f"Attempted to save to: {output_path}")
                return None
            except Exception as e:
                print(f"Error saving {name}.csv: {str(e)}")
                return None

        return splits

    except Exception as e:
        print(f"Error processing data: {str(e)}")
        return None
    
def verify_data_splits():
    """
    Verify data split integrity
    File Location: /src/data/preprocess.py
    """
    # Load all splits
    splits = {
        'train': {'X': pd.read_csv('X_train.csv'), 'y': pd.read_csv('y_train.csv')},
        'val': {'X': pd.read_csv('X_val.csv'), 'y': pd.read_csv('y_val.csv')},
        'test': {'X': pd.read_csv('X_test.csv'), 'y': pd.read_csv('y_test.csv')}
    }
    
    for split_name, data in splits.items():
        print(f"\nVerifying {split_name} split:")
        
        # Check dimensions
        print(f"X shape: {data['X'].shape}")
        print(f"y shape: {data['y'].shape}")
        
        # Verify fixture dimensions
        for fixture in ['Toilet', 'Sink', 'Bathtub']:
            width_col = f'{fixture}_Width'
            depth_col = f'{fixture}_Depth'
            if width_col in data['y'].columns and depth_col in data['y'].columns:
                print(f"\n{fixture} dimensions:")
                print(f"Width range: {data['y'][width_col].min():.1f} to {data['y'][width_col].max():.1f}")
                print(f"Depth range: {data['y'][depth_col].min():.1f} to {data['y'][depth_col].max():.1f}")

if __name__ == "__main__":
    # Update these paths according to your setup
    input_filepath = "H:\\Shared drives\\AI Design Tool\\00-PG_folder\\03-Furniture AI Model\\Data\preprocessed_dataset.csv"
    output_directory = "H:\\Shared drives\\AI Design Tool\\00-PG_folder\\03-Furniture AI Model\\Data\\preprocessed"
    
    try:
        splits = load_and_split_data(input_filepath, output_directory)
        if splits is not None:
            print("\nData processing completed successfully!")
        else:
            print("\nData processing failed!")
    except Exception as e:
        print(f"\nError in main execution: {str(e)}")