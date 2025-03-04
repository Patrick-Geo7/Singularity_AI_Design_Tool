import pandas as pd
import numpy as np

# Load the latest dataset
file_path = "/media/patrick/Patrick/Singularity_AI_Design_Tool/Data/augmented_bathroom_dataset_relaxed.csv"
df = pd.read_csv(file_path)


def generate_diverse_room_shapes():
    """ Generate diverse room shapes with realistic dimensions matching the initial dataset."""
    return "Rectangle", np.random.randint(60, 102), np.random.randint(60, 102)  # Width x Length (inches)


def adjust_for_door_positions(sample):
    """ Adjust fixture placement based on door position. """
    door_x, door_y = sample["Door_X_Position"], sample["Door_Y_Position"]
    door_width = sample["Door_Width"]

    for fixture in ["Toilet", "Sink", "Bathtub"]:
        x_col, y_col = f"{fixture}_X", f"{fixture}_Y"
        if x_col in sample and y_col in sample:
            if door_x - door_width < sample[x_col] < door_x + door_width:
                sample[x_col] += np.random.uniform(5, 10)  # Shift away from the door
    return sample


def enforce_rotation_adjustment(sample):
    """ Adjust fixture dimensions based on rotation. """
    for fixture in ["Toilet", "Sink", "Bathtub"]:
        width_col, depth_col = f"{fixture}_Width", f"{fixture}_Depth"
        rot_col = f"{fixture}_Rotation"

        if rot_col in sample and sample[rot_col] in [90, 270]:
            sample[width_col], sample[depth_col] = sample[depth_col], sample[width_col]  # Swap dimensions
    return sample


def enforce_no_overlap(sample):
    """ Adjust fixture positions to prevent overlap. """
    fixtures = ["Toilet", "Sink", "Bathtub"]
    fixture_positions = {}

    for fixture in fixtures:
        x_col, y_col = f"{fixture}_X", f"{fixture}_Y"
        width_col, depth_col = f"{fixture}_Width", f"{fixture}_Depth"

        if x_col in sample and y_col in sample:
            fixture_positions[fixture] = [
                sample[x_col],
                sample[y_col],
                sample[x_col] + sample[width_col],
                sample[y_col] + sample[depth_col]
            ]

    for f1 in fixtures:
        if f1 in fixture_positions:
            x1, y1, x2, y2 = fixture_positions[f1]

            for f2 in fixtures:
                if f1 != f2 and f2 in fixture_positions:
                    x1b, y1b, x2b, y2b = fixture_positions[f2]

                    if not (x2 <= x1b or x1 >= x2b or y2 <= y1b or y1 >= y2b):
                        shift_x, shift_y = np.random.uniform(2, 5), np.random.uniform(2, 5)
                        if x2b + shift_x < sample["Room_Width"]:
                            sample[f"{f2}_X"] += shift_x
                        if y2b + shift_y < sample["Room_Length"]:
                            sample[f"{f2}_Y"] += shift_y

    return sample


def augment_advanced_dataset(df, num_samples=50000):
    """ Generate 50,000 samples with diverse layouts and better constraints matching the initial dataset."""
    augmented_samples = []

    for _ in range(num_samples):
        sample = df.sample(n=1, replace=True).iloc[0].copy()

        shape_type, room_width, room_length = generate_diverse_room_shapes()
        sample["Room_Shape"] = shape_type
        sample["Room_Width"] = room_width
        sample["Room_Length"] = room_length

        sample = adjust_for_door_positions(sample)
        sample = enforce_rotation_adjustment(sample)
        sample = enforce_no_overlap(sample)

        augmented_samples.append(sample)

    return pd.DataFrame(augmented_samples)


# Generate the final, improved dataset
augmented_advanced_df = augment_advanced_dataset(df, num_samples=50000)

# Save dataset
augmented_advanced_file_path = "/media/patrick/Patrick/Singularity_AI_Design_Tool/Data/augmented_bathroom_dataset_advanced.csv"
augmented_advanced_df.to_csv(augmented_advanced_file_path, index=False)
