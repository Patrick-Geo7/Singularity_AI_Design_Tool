import pandas as pd
import numpy as np

# Load the latest dataset
file_path = "/media/patrick/Patrick/Singularity_AI_Design_Tool/Data/augmented_bathroom_dataset_relaxed.csv"
df = pd.read_csv(file_path)

def generate_diverse_room_shapes():
    """ Generate diverse room shapes including rectangular and L-shaped layouts. """
    room_types = ["Rectangle", "L-Shape"]
    shape_type = np.random.choice(room_types, p=[0.7, 0.3])  # 70% rectangular, 30% L-shape

    if shape_type == "Rectangle":
        return shape_type, np.random.randint(250, 500), np.random.randint(250, 500)  # Width x Length (cm)
    else:  # L-Shape
        main_length = np.random.randint(300, 500)
        main_width = np.random.randint(250, 400)
        cutout_length = np.random.randint(100, main_length // 2)
        cutout_width = np.random.randint(100, main_width // 2)
        return shape_type, (main_length, cutout_length), (main_width, cutout_width)

def adjust_for_door_positions(sample):
    """ Adjust fixture placement based on door position. """
    door_x, door_y = sample["Door_X_Position"], sample["Door_Y_Position"]
    door_width = sample["Door_Width"]

    # Ensure fixtures are not placed directly in front of the door
    for fixture in ["Toilet", "Sink", "Bathtub"]:
        x_col, y_col = f"{fixture}_X_Position", f"{fixture}_Y_Position"
        if x_col in sample and y_col in sample:
            if door_x - door_width < sample[x_col] < door_x + door_width:
                sample[x_col] += np.random.uniform(20, 40)  # Shift away from the door
    return sample

def heuristic_layout_score(sample):
    """ Assigns a heuristic score based on practical layout conditions. """
    score = 100  # Start with a base score

    # Penalize fixtures that are too close to the door
    for fixture in ["Toilet", "Sink", "Bathtub"]:
        x_col, y_col = f"{fixture}_X_Position", f"{fixture}_Y_Position"
        if x_col in sample and y_col in sample:
            distance_to_door = abs(sample[x_col] - sample["Door_X_Position"])
            if distance_to_door < 50:
                score -= 20  # Penalize for being too close

    # Bonus for well-spaced layouts (non-cramped rooms)
    if sample["Room_Length"] * sample["Room_Width"] > 100000:  # Large rooms
        score += 10

    return max(0, score)  # Ensure score is non-negative

def enforce_no_overlap(sample):
    """ Adjusts fixture positions to ensure no overlapping at the end. """
    fixtures = ["Toilet", "Sink", "Bathtub"]
    fixture_positions = {}

    # Store fixture boundaries (bottom-left and top-right)
    for fixture in fixtures:
        x_col, y_col = f"{fixture}_X_Position", f"{fixture}_Y_Position"
        width_col, depth_col = f"{fixture}_Width", f"{fixture}_Depth"

        if x_col in sample and y_col in sample:
            fixture_positions[fixture] = [
                sample[x_col],  # x1 (bottom-left)
                sample[y_col],  # y1 (bottom-left)
                sample[x_col] + sample[width_col],  # x2 (top-right)
                sample[y_col] + sample[depth_col],  # y2 (top-right)
            ]

    # Check for overlaps and adjust positions
    for f1 in fixtures:
        if f1 in fixture_positions:
            x1, y1, x2, y2 = fixture_positions[f1]

            for f2 in fixtures:
                if f1 != f2 and f2 in fixture_positions:
                    x1b, y1b, x2b, y2b = fixture_positions[f2]

                    # Check overlap condition
                    if not (x2 <= x1b or x1 >= x2b or y2 <= y1b or y1 >= y2b):
                        # If overlapping, shift the second fixture slightly
                        shift_x, shift_y = np.random.uniform(5, 15), np.random.uniform(5, 15)
                        if x2b + shift_x < sample["Room_Length"]:  # Ensure within bounds
                            sample[f"{f2}_X_Position"] += shift_x
                        if y2b + shift_y < sample["Room_Width"]:  # Ensure within bounds
                            sample[f"{f2}_Y_Position"] += shift_y

    return sample

def augment_advanced_dataset(df, num_samples=50000):
    """ Generate 50,000 samples with diverse layouts and better constraints. """
    augmented_samples = []

    for _ in range(num_samples):
        sample = df.sample(n=1, replace=True).iloc[0].copy()

        # Generate diverse room shapes
        shape_type, room_length, room_width = generate_diverse_room_shapes()
        sample["Shape_Type"] = shape_type
        sample["Room_Length"] = room_length if isinstance(room_length, int) else sum(room_length)
        sample["Room_Width"] = room_width if isinstance(room_width, int) else sum(room_width)

        # Apply door-aware fixture placement
        sample = adjust_for_door_positions(sample)

        # Ensure no overlapping at the end
        sample = enforce_no_overlap(sample)

        # Assign heuristic score for layout quality
        sample["Layout_Score"] = heuristic_layout_score(sample)

        augmented_samples.append(sample)

    return pd.DataFrame(augmented_samples)

# Generate the final, improved dataset
augmented_advanced_df = augment_advanced_dataset(df, num_samples=50000)

# Save dataset
augmented_advanced_file_path = "/media/patrick/Patrick/Singularity_AI_Design_Tool/Data/augmented_bathroom_dataset_advanced.csv"
augmented_advanced_df.to_csv(augmented_advanced_file_path, index=False)

