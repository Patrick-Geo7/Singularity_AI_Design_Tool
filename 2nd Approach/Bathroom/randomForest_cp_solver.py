from ortools.sat.python import cp_model
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Load dataset
file_path = "H:/Shared drives/AI Design Tool/00-PG_folder/03-Furniture AI Model/Data/preprocessed_dataset.csv"
df = pd.read_csv(file_path)

# Define input features and target variables
input_features = ['Room_Length', 'Room_Width', 'Door_X_Position', 'Door_Y_Position', 'Door_Width']
target_features = ['Toilet_X_Position', 'Toilet_Y_Position','Toilet_Width','Toilet_Depth', 'Toilet_Rotation',
                   'Sink_X_Position', 'Sink_Y_Position','Sink_Width','Sink_Depth', 'Sink_Rotation',
                   'Bathtub_X_Position', 'Bathtub_Y_Position','Bathtub_Width','Sink_Depth','Bathtub_Rotation']

# Split data
X = df[input_features]
y = df[target_features]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# Train ML model
ml_model = RandomForestRegressor(n_estimators=1500, random_state=10)
ml_model.fit(X_train, y_train)

# Function to visualize ML predictions
def visualize_ml_predictions(room_width, room_length, door_x, door_y, door_width, fixtures, ml_predictions):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, room_width)
    ax.set_ylim(0, room_length)
    ax.set_title("ML Predicted Bathroom Layout (Before Constraints)")
    
    # Draw Room Boundary
    ax.add_patch(patches.Rectangle((0, 0), room_width, room_length, edgecolor='black', facecolor='none', lw=2))
    
    # Draw Door
    ax.add_patch(patches.Rectangle((door_x, door_y), door_width, 5, edgecolor='blue', facecolor='blue', alpha=0.5))
    
    # Draw Fixtures based on ML predictions
    colors = {'Toilet': 'red', 'Sink': 'green', 'Bathtub': 'purple'}
    for name, fixture in fixtures.items():
        x = ml_predictions.get(f'{name}_X_Position', 0)
        y = ml_predictions.get(f'{name}_Y_Position', 0)
        rotation = ml_predictions.get(f'{name}_Rotation', 0)
        
        # Adjust width and depth based on rotation
        if rotation in [1, 3]:  # 90° or 270°
            w, d = fixture['depth'], fixture['width']
        else:  # 0° or 180°
            w, d = fixture['width'], fixture['depth']
        
        ax.add_patch(patches.Rectangle((x, y), w, d, edgecolor=colors[name], facecolor=colors[name], alpha=0.5))
        ax.text(x + w/2, y + d/2, name, color='white', ha='center', va='center', fontsize=10)
    
    plt.xlabel("Width")
    plt.ylabel("Length")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

# Define function to optimize bathroom layout
def optimize_bathroom_layout(room_width, room_length, door_x, door_y, door_width, fixtures, ml_predictions):
    model = cp_model.CpModel()
    
    # Decision Variables (X, Y, Rotation)
    x_vars = {}
    y_vars = {}
    rotation_vars = {}
    
    for name, fixture in fixtures.items():
        x_vars[name] = model.NewIntVar(0, room_width, f'{name}_x')
        y_vars[name] = model.NewIntVar(0, room_length, f'{name}_y')
        rotation_vars[name] = model.NewIntVar(0, 3, f'{name}_rotation')
        
        # Set initial values from ML model as hints
        model.AddHint(x_vars[name], int(ml_predictions.get(f'{name}_X_Position', 0)))
        model.AddHint(y_vars[name], int(ml_predictions.get(f'{name}_Y_Position', 0)))
        model.AddHint(rotation_vars[name], int(ml_predictions.get(f'{name}_Rotation', 0)))
    
    # Constraints: Fixtures must be attached to a wall and respect rotation rules
    for name, fixture in fixtures.items():
        attached_to_left = model.NewBoolVar(f"{name}_attached_to_left")
        attached_to_right = model.NewBoolVar(f"{name}_attached_to_right")
        attached_to_bottom = model.NewBoolVar(f"{name}_attached_to_bottom")
        attached_to_top = model.NewBoolVar(f"{name}_attached_to_top")

        model.Add(x_vars[name] == 0).OnlyEnforceIf(attached_to_left)
        model.Add(x_vars[name] + fixture['width'] == room_width).OnlyEnforceIf(attached_to_right)
        model.Add(y_vars[name] == 0).OnlyEnforceIf(attached_to_bottom)
        model.Add(y_vars[name] + fixture['depth'] == room_length).OnlyEnforceIf(attached_to_top)

        model.AddBoolOr([attached_to_left, attached_to_right, attached_to_bottom, attached_to_top])
    
    # Ensure fixtures stay within room boundaries
    for name, fixture in fixtures.items():
        model.Add(x_vars[name] + fixture['width'] <= room_width)
        model.Add(y_vars[name] + fixture['depth'] <= room_length)
    
    # Respect door clearance
    door_clearance = 30  # Minimum clearance in front of the door
    for name, fixture in fixtures.items():
        # Add clearance constraints for X and Y positions
        diff_x = model.NewIntVar(-room_width, room_width, f"{name}_diff_x")
        diff_y = model.NewIntVar(-room_length, room_length, f"{name}_diff_y")
        
        # Calculate differences
        model.Add(diff_x == x_vars[name] - door_x)
        model.Add(diff_y == y_vars[name] - door_y)
        
        # Calculate absolute differences
        abs_diff_x = model.NewIntVar(0, room_width, f"{name}_abs_diff_x")
        abs_diff_y = model.NewIntVar(0, room_length, f"{name}_abs_diff_y")
        
        model.AddAbsEquality(abs_diff_x, diff_x)  # Correct usage of AddAbsEquality
        model.AddAbsEquality(abs_diff_y, diff_y)  # Correct usage of AddAbsEquality
        
        # Enforce minimum clearance
        model.Add(abs_diff_x >= door_clearance)
        model.Add(abs_diff_y >= door_clearance)
    
    # No Overlapping Constraint
    for name1, fixture1 in fixtures.items():
        for name2, fixture2 in fixtures.items():
            if name1 != name2:
                no_overlap_x = model.NewBoolVar(f"{name1}_{name2}_no_overlap_x")
                no_overlap_y = model.NewBoolVar(f"{name1}_{name2}_no_overlap_y")
                model.Add(x_vars[name1] + fixture1['width'] <= x_vars[name2]).OnlyEnforceIf(no_overlap_x)
                model.Add(y_vars[name1] + fixture1['depth'] <= y_vars[name2]).OnlyEnforceIf(no_overlap_y)
                model.AddBoolOr([no_overlap_x, no_overlap_y])
    
    # Solve the model
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    
    if status == cp_model.FEASIBLE or status == cp_model.OPTIMAL:
        result = {name: {'X': solver.Value(x_vars[name]), 'Y': solver.Value(y_vars[name]), 'Rotation': solver.Value(rotation_vars[name])} for name in fixtures}
        return result
    else:
        return None

# Function to visualize the bathroom layout
def visualize_layout(room_width, room_length, door_x, door_y, door_width, fixtures, solution):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, room_width)
    ax.set_ylim(0, room_length)
    ax.set_title("Optimized Bathroom Layout (After Constraints)")
    
    # Draw Room Boundary
    ax.add_patch(patches.Rectangle((0, 0), room_width, room_length, edgecolor='black', facecolor='none', lw=2))
    
    # Draw Door
    ax.add_patch(patches.Rectangle((door_x, door_y), door_width, 5, edgecolor='blue', facecolor='blue', alpha=0.5))
    
    # Draw Fixtures
    colors = {'Toilet': 'red', 'Sink': 'green', 'Bathtub': 'purple'}
    for name, values in solution.items():
        x, y = values['X'], values['Y']
        rotation = values['Rotation']
        
        # Adjust width and depth based on rotation
        if rotation in [1, 3]:  # 90° or 270°
            w, d = fixtures[name]['depth'], fixtures[name]['width']
        else:  # 0° or 180°
            w, d = fixtures[name]['width'], fixtures[name]['depth']
        
        ax.add_patch(patches.Rectangle((x, y), w, d, edgecolor=colors[name], facecolor=colors[name], alpha=0.5))
        ax.text(x + w/2, y + d/2, name, color='white', ha='center', va='center', fontsize=10)
    
    plt.xlabel("Width")
    plt.ylabel("Length")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

# Run optimization for test data
for index, row in X_test.iterrows():
    room_width = row['Room_Width']
    room_length = row['Room_Length']
    door_x = row['Door_X_Position']
    door_y = row['Door_Y_Position']
    door_width = row['Door_Width']
    
    fixtures = {
        "Toilet": {"width": 19, "depth": 28},
        "Sink": {"width": 30, "depth": 20},
        "Bathtub": {"width": 30, "depth": 60}
    }
    
    # Predict initial layout using ML model
    ml_input = pd.DataFrame([row], columns=input_features)
    ml_pred = ml_model.predict(ml_input)[0]
    ml_predictions = {key: ml_pred[i] for i, key in enumerate(target_features)}
    
    # Visualize ML predictions (before constraints)
    visualize_ml_predictions(room_width, room_length, door_x, door_y, door_width, fixtures, ml_predictions)
    
    # Run constraint solver
    solution = optimize_bathroom_layout(room_width, room_length, door_x, door_y, door_width, fixtures, ml_predictions)
    if solution:
        print(f"Bathroom {index} solution:", solution)
        # Visualize optimized layout (after constraints)
        visualize_layout(room_width, room_length, door_x, door_y, door_width, fixtures, solution)
    else:
        print(f"Bathroom {index} solution is not found")