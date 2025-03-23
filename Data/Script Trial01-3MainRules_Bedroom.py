import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from matplotlib.transforms import Affine2D
from matplotlib import image as mpimg
from shapely.geometry import box
from shapely.affinity import rotate
import math
import os
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend for better file saving

# Constants
ROOM_WIDTH = 120  # inches (adjust as needed for bedroom)
ROOM_HEIGHT = 160  # inches (adjust as needed for bedroom)
DOOR_WIDTH = 30
WINDOW_WIDTH = 30
WALL_THICKNESS = 5
fixtures = {
    'single_bed': ['Data/Assets/2d_Images/New Assets/Single Bed.png', 'Data/Assets/2d_Images/New Assets/Single Bed with Back.png'],
    'double_bed': ['Data/Assets/2d_Images/New Assets/Double Bed.png', 'Data/Assets/2d_Images/New Assets/Double Bed with Back.png'],
    'commode': ['Data/Assets/2d_Images/New Assets/Double Steel.png', 'Data/Assets/2d_Images/New Assets/Double Steel with Drawers.png','Data/Assets/2d_Images/New Assets/Double Lifted.png','Data/Assets/2d_Images/New Assets/4 Panels with Drawers.png'],
    'wardrobe': ['Data/Assets/2d_Images/New Assets/Wardrobe with Loft.png', 'Data/Assets/2d_Images/New Assets/White Single Dresser.png','Data/Assets/2d_Images/New Assets/Wooden Single Dresser.png']
}
NUM_LAYOUTS = 5000

# Fixture dimensions (width, height)
SINGLE_BED = (36, 75)  # Single bed dimensions (width, length)
DOUBLE_BED = (54, 75)  # Double bed dimensions
COMMODE = (18, 22)  # Commode dimensions
WARDROBE = (40, 60)  # Wardrobe dimensions


# Function to check if a fixture is within room boundaries
def is_within_boundaries(x, y, width, height, angle):
    fixture = box(x, y, x + width, y + height)
    center_x, center_y = x + width / 2, y + height / 2
    fixture = rotate(fixture, angle, origin=(center_x, center_y))
    room = box(0, 0, ROOM_WIDTH, ROOM_HEIGHT)
    return room.contains(fixture)


# Function to check if two fixtures overlap
def do_fixtures_overlap(fixture1, fixture2):
    return fixture1.intersects(fixture2) and not fixture1.touches(fixture2)


# Function to create a fixture polygon
from shapely.geometry import Polygon
from shapely.affinity import rotate


def create_fixture_polygon(x, y, width, height, angle):
    center_x = x + width / 2
    center_y = y + height / 2
    fixture = Polygon([(x, y), (x + width, y), (x + width, y + height), (x, y + height)])
    rotated_fixture = rotate(fixture, angle, origin=(center_x, center_y), use_radians=False)
    return rotated_fixture


def is_door_position_valid(door_x, door_y, fixtures):
    for fixture in fixtures:
        fx, fy, _, fw, fh, _ = fixture
        if (door_x >= fx and door_x <= fx + fw) and (door_y >= fy and door_y <= fy + fh):
            return False
        if (door_x + DOOR_WIDTH >= fx and door_x <= fx + fw) and (door_y + 30 >= fy and door_y <= fy + fh):
            return False
    return True


# Function to generate a valid layout
def generate_valid_layout():
    max_attempts = 1000
    attempts = 0
    possible_angles = [0, 90, 180, 270]

    while attempts < max_attempts:
        # Randomly select bedroom type (single or double)
        bedroom_type = random.choice(['single', 'double'])

        # Randomly select fixture angles
        bed_angle = random.choice(possible_angles)
        commode_angle = random.choice(possible_angles)
        wardrobe_angle = random.choice(possible_angles)

        # Adjust fixture dimensions based on bedroom type
        bed_w, bed_h = SINGLE_BED if bedroom_type == 'single' else DOUBLE_BED
        commode_w, commode_h = COMMODE
        wardrobe_w, wardrobe_h = WARDROBE

        # Position fixtures
        bed_x = random.randint(0, ROOM_WIDTH - bed_w)
        bed_y = random.randint(0, ROOM_HEIGHT - bed_h)
        commode_x = random.randint(0, ROOM_WIDTH - commode_w)
        commode_y = random.randint(0, ROOM_HEIGHT - commode_h)
        wardrobe_x = random.randint(0, ROOM_WIDTH - wardrobe_w)
        wardrobe_y = random.randint(0, ROOM_HEIGHT - wardrobe_h)

        # Check boundaries
        if not is_within_boundaries(bed_x, bed_y, bed_w, bed_h, bed_angle):
            attempts += 1
            continue
        if not is_within_boundaries(commode_x, commode_y, commode_w, commode_h, commode_angle):
            attempts += 1
            continue
        if not is_within_boundaries(wardrobe_x, wardrobe_y, wardrobe_w, wardrobe_h, wardrobe_angle):
            attempts += 1
            continue

        # Create fixture polygons
        bed_poly = create_fixture_polygon(bed_x, bed_y, bed_w, bed_h, bed_angle)
        commode_poly = create_fixture_polygon(commode_x, commode_y, commode_w, commode_h, commode_angle)
        wardrobe_poly = create_fixture_polygon(wardrobe_x, wardrobe_y, wardrobe_w, wardrobe_h, wardrobe_angle)

        if do_fixtures_overlap(bed_poly, commode_poly) or do_fixtures_overlap(bed_poly, wardrobe_poly) or \
                do_fixtures_overlap(commode_poly, wardrobe_poly):
            attempts += 1
            continue

        # Choose door position
        door_wall = random.choice(['bottom', 'left', 'top', 'right'])
        if door_wall == 'bottom':
            door_x = random.randint(0, ROOM_WIDTH - DOOR_WIDTH)
            door_y = 0
        elif door_wall == 'top':
            door_x = random.randint(0, ROOM_WIDTH - DOOR_WIDTH)
            door_y = ROOM_HEIGHT
        elif door_wall == 'left':
            door_x = 0
            door_y = random.randint(0, ROOM_HEIGHT - DOOR_WIDTH)
        else:
            door_x = ROOM_WIDTH
            door_y = random.randint(0, ROOM_HEIGHT - DOOR_WIDTH)

        # Validate door position
        if not is_door_position_valid(door_x, door_y, [(bed_x, bed_y, 0, bed_w, bed_h, bed_angle),
                                                       (commode_x, commode_y, 0, commode_w, commode_h, commode_angle),
                                                       (wardrobe_x, wardrobe_y, 0, wardrobe_w, wardrobe_h,
                                                        wardrobe_angle)]):
            attempts += 1
            continue

        return {
            'bed': (bed_x, bed_y, 0, bed_w, bed_h, bed_angle),
            'commode': (commode_x, commode_y, 0, commode_w, commode_h, commode_angle),
            'wardrobe': (wardrobe_x, wardrobe_y, 0, wardrobe_w, wardrobe_h, wardrobe_angle),
            'door': (door_x, door_y, 0, DOOR_WIDTH, 'open')
        }

    return None


# Function to visualize layout
def visualize_layout(layout, layout_num):
    fig, ax = plt.subplots(figsize=(8, 9))

    # Draw room
    room = patches.Rectangle((0, 0), ROOM_WIDTH, ROOM_HEIGHT,
                             linewidth=2, edgecolor='black', facecolor='none')
    ax.add_patch(room)

    # Draw fixtures using images
    for fixture_name, image_paths in fixtures.items():
        if fixture_name in layout:
            x, y, z, width, height, angle = layout[fixture_name]
            image_path = random.choice(image_paths)  # Select a random image
            try:
                img = mpimg.imread(image_path)
                if angle == 90:
                    img = np.rot90(img, k=1)
                elif angle == 180:
                    img = np.rot90(img, k=2)
                elif angle == 270:
                    img = np.rot90(img, k=3)
                ax.imshow(img, extent=(x, x + width, y, y + height), aspect='auto', alpha=1.0)
            except FileNotFoundError:
                print(f"Error: Image not found for {fixture_name} at {image_path}")

    # Set limits and title
    ax.set_xlim(0, ROOM_WIDTH)
    ax.set_ylim(0, ROOM_HEIGHT)
    ax.set_aspect('equal')
    ax.set_title(f'Bedroom Layout #{layout_num + 1}')
    ax.set_xlabel('Width (inches)')
    ax.set_ylabel('Height (inches)')

    plt.tight_layout()

    # Save the layout image
    output_directory = os.path.join(os.getcwd(), "Generated_Bedroom_Layouts")
    os.makedirs(output_directory, exist_ok=True)
    output_file = os.path.join(output_directory, f'bedroom_layout_{layout_num + 1}.png')

    try:
        plt.savefig(output_file, dpi=100, bbox_inches='tight')
        print(f"Successfully saved layout {layout_num + 1}")
    except Exception as e:
        print(f"Error saving layout: {e}")

    plt.close(fig)


# Generate and visualize layouts
def generate_layouts(num_layouts):
    unique_layouts = []
    attempts = 0
    max_attempts = num_layouts * 10

    print(f"Generating {num_layouts} unique bedroom layouts...")

    while len(unique_layouts) < num_layouts and attempts < max_attempts:
        layout = generate_valid_layout()
        attempts += 1

        if layout is not None:
            unique_layouts.append(layout)
            print(f"Layout {len(unique_layouts)}/{num_layouts} generated")

    # Visualize all unique layouts
    for i, layout in enumerate(unique_layouts):
        visualize_layout(layout, i)

    return unique_layouts


# Main execution
if __name__ == "__main__":
    random.seed(84)
    np.random.seed(84)

    # Generate layouts
    layouts = generate_layouts(NUM_LAYOUTS)

    print(f"Successfully generated {len(layouts)} bedroom layouts")
    print("Layout images have been saved as bedroom_layout_X.png files")
