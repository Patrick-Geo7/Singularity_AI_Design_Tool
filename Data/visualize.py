import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random


def plot_bathroom(sample):
    fig, ax = plt.subplots(figsize=(6, 6))

    # Draw room boundaries
    room_length, room_width = sample["Room_Length"], sample["Room_Width"]
    ax.set_xlim(0, room_length)
    ax.set_ylim(0, room_width)
    ax.set_title("Bathroom Layout")
    ax.set_xlabel("Length (cm)")
    ax.set_ylabel("Width (cm)")

    # Draw room rectangle
    room_rect = patches.Rectangle((0, 0), room_length, room_width, linewidth=2, edgecolor='black', facecolor='none')
    ax.add_patch(room_rect)

    # Draw fixtures as rectangles
    fixtures = {"Toilet": "r", "Sink": "b", "Bathtub": "g"}
    fixture_sizes = {"Toilet": (40, 40), "Sink": (30, 30), "Bathtub": (150, 70)}  # Example sizes

    for fixture, color in fixtures.items():
        x_col, y_col = f"{fixture}_X_Position", f"{fixture}_Y_Position"

        if x_col in sample and y_col in sample:
            width, height = fixture_sizes.get(fixture, (50, 50))
            rect = patches.Rectangle((sample[x_col], sample[y_col]), width, height, linewidth=2, edgecolor=color,
                                     facecolor=color, alpha=0.5)
            ax.add_patch(rect)
            ax.text(sample[x_col] + 5, sample[y_col] + height / 2, fixture, fontsize=9, color='white', weight='bold')

    ax.legend(handles=[patches.Patch(color=color, label=fixture) for fixture, color in fixtures.items()])
    plt.grid()
    plt.show()


# Load dataset
file_path = "augmented_bathroom_dataset_relaxed.csv"  # Update path if needed
df = pd.read_csv(file_path)

# Randomly visualize 5 samples
for _ in range(5):
    sample = df.sample(n=1).iloc[0]
    plot_bathroom(sample)
