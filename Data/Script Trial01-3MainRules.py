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
ROOM_WIDTH = 96  # inches
ROOM_HEIGHT = 108  # inches
DOOR_WIDTH = 30
WINDOW_WIDTH = 30
WALL_THICKNESS = 5
fixtures = {'toilet' : 'Data/Assets/2d_Images/toilet.png',
            'sink' : 'Data/Assets/2d_Images/sink.png',
            'bathtub' : 'Data/Assets/2d_Images/bathtub.png'}
NUM_LAYOUTS = 5000

# Fixture dimensions (width, height)
BATHTUB = (60, 30)  # flipped to match conventional orientation
TOILET = (19, 28)
SINK = (30, 20)

# Function to draw a door for the room


# Function to check if a fixture is within room boundaries
def is_within_boundaries(x, y, width, height, angle):
    # Create the fixture polygon
    fixture = box(x, y, x + width, y + height)
    # Rotate the fixture around its center
    center_x, center_y = x + width/2, y + height/2
    fixture = rotate(fixture, angle, origin=(center_x, center_y))
    
    # Create the room polygon
    room = box(0, 0, ROOM_WIDTH, ROOM_HEIGHT)
    
    # Check if the fixture is within the room
    return room.contains(fixture)

# Function to check if two fixtures overlap
def do_fixtures_overlap(fixture1, fixture2):
    return fixture1.intersects(fixture2) and not fixture1.touches(fixture2)

# Function to create a fixture polygon
from shapely.geometry import Polygon
from shapely.affinity import rotate


def create_fixture_polygon(x, y, width, height, angle):
    """
    Creates a rotated fixture polygon around its center.

    Parameters:
    - x, y: The top-left corner of the fixture before rotation.
    - width, height: The dimensions of the fixture.
    - angle: Rotation angle in degrees.

    Returns:
    - A rotated Shapely Polygon.
    """
    # Compute the fixture's center
    center_x = x + width / 2
    center_y = y + height / 2

    # Define the unrotated rectangle (assuming top-left as (x, y))
    fixture = Polygon([
        (x, y),  # Top-left
        (x + width, y),  # Top-right
        (x + width, y + height),  # Bottom-right
        (x, y + height)  # Bottom-left
    ])

    # Rotate the polygon around its center
    rotated_fixture = rotate(fixture, angle, origin=(center_x, center_y), use_radians=False)

    return rotated_fixture


# Function to generate a valid layout
def generate_valid_layout():
    max_attempts = 1000
    attempts = 0
    
    # Define possible angles in 90-degree increments
    possible_angles = [0, 90, 180, 270]
    
    while attempts < max_attempts:
        # Choose angles in 90-degree increments for all fixtures
        bathtub_angle = random.choice(possible_angles)
        toilet_angle = random.choice(possible_angles)
        sink_angle = random.choice(possible_angles)
        
        # Adjust dimensions based on rotation
        bathtub_w, bathtub_h = (BATHTUB[1], BATHTUB[0]) if bathtub_angle in [90, 270] else BATHTUB
        toilet_w, toilet_h = (TOILET[1], TOILET[0]) if toilet_angle in [90, 270] else TOILET
        sink_w, sink_h = (SINK[1], SINK[0]) if sink_angle in [90, 270] else SINK
        
        # For wall alignment, we'll randomly choose which wall to align to
        # and then position the fixture along that wall

        # Bathtub alignment
        if bathtub_angle == 0:  # On bottom wall, facing up
            bathtub_x = random.randint(0, ROOM_WIDTH - bathtub_w)
            bathtub_y = ROOM_HEIGHT - bathtub_h  # Align to bottom wall
        elif bathtub_angle == 90:  # On left wall, facing right
            bathtub_x = 0
            bathtub_y = random.randint(0, ROOM_HEIGHT - bathtub_h)
        elif bathtub_angle == 180:  # On top wall, facing down
            bathtub_x = random.randint(0, ROOM_WIDTH - bathtub_w)
            bathtub_y = 0  # Align to top wall
        elif bathtub_angle == 270:  # On right wall, facing left
            bathtub_x = ROOM_WIDTH - bathtub_w
            bathtub_y = random.randint(0, ROOM_HEIGHT - bathtub_h)

        bathtub_z = 0

        # Toilet alignment
        if toilet_angle == 0:
            toilet_x = random.randint(0, ROOM_WIDTH - toilet_w)
            toilet_y = ROOM_HEIGHT - toilet_h
        elif toilet_angle == 90:
            toilet_x = 0
            toilet_y = random.randint(0, ROOM_HEIGHT - toilet_h)
        elif toilet_angle == 180:
            toilet_x = random.randint(0, ROOM_WIDTH - toilet_w)
            toilet_y = 0
        elif toilet_angle == 270:
            toilet_x = ROOM_WIDTH - toilet_w
            toilet_y = random.randint(0, ROOM_HEIGHT - toilet_h)
        toilet_z = 0

        # Sink alignment
        if sink_angle == 0:
            sink_x = random.randint(0, ROOM_WIDTH - sink_w)
            sink_y = ROOM_HEIGHT - sink_h
        elif sink_angle == 90:
            sink_x = 0
            sink_y = random.randint(0, ROOM_HEIGHT - sink_h)
        elif sink_angle == 180:
            sink_x = random.randint(0,ROOM_HEIGHT - sink_w)
            sink_y = 0
        else:  # Default (180 degrees)
            sink_x = ROOM_WIDTH - sink_w
            sink_y = random.randint(0, ROOM_HEIGHT - sink_h)
        sink_z = 0
        # Check boundaries
        if not is_within_boundaries(bathtub_x, bathtub_y, bathtub_w, bathtub_h, bathtub_angle):
            attempts += 1
            continue
            
        if not is_within_boundaries(toilet_x, toilet_y, toilet_w, toilet_h, toilet_angle):
            attempts += 1
            continue
            
        if not is_within_boundaries(sink_x, sink_y, sink_w, sink_h, sink_angle):
            attempts += 1
            continue
        
        # Create fixture polygons
        bathtub_poly = create_fixture_polygon(bathtub_x, bathtub_y, bathtub_w, bathtub_h, bathtub_angle)
        toilet_poly = create_fixture_polygon(toilet_x, toilet_y, toilet_w+18, toilet_h+36, toilet_angle)
        sink_poly = create_fixture_polygon(sink_x, sink_y, sink_w, sink_h+30, sink_angle)
        
        # Check overlaps
        if do_fixtures_overlap(bathtub_poly, toilet_poly) or \
           do_fixtures_overlap(bathtub_poly, sink_poly) or \
           do_fixtures_overlap(toilet_poly, sink_poly):
            attempts += 1
            continue
            
        # Valid layout found
        return {
            'bathtub': (bathtub_x, bathtub_y, bathtub_z, bathtub_w, bathtub_h, bathtub_angle),
            'toilet': (toilet_x, toilet_y, toilet_z, toilet_w, toilet_h, toilet_angle),
            'sink': (sink_x, sink_y, sink_z, sink_w, sink_h, sink_angle)
        }
        
    # If we reach here, we couldn't find a valid layout
    return None

# Function to draw an arrow indicating fixture orientation
def draw_orientation_arrow(ax, x, y, width, height, angle):
    # Calculate center of the fixture
    center_x = x + width/2
    center_y = y + height/2
    
    # Calculate arrow length (proportional to fixture size)
    arrow_length = min(width, height) * 0.3
    
    # Calculate arrow endpoint based on angle
    if angle == 0:  # pointing right
        end_x = center_x
        end_y = center_y + arrow_length
    elif angle == 90:  # pointing up
        end_x = center_x + arrow_length
        end_y = center_y
    elif angle == 180:  # pointing left
        end_x = center_x
        end_y = center_y - arrow_length
    else:  # angle == 270, pointing down
        end_x = center_x - arrow_length
        end_y = center_y
    
    # Draw the arrow
    ax.arrow(center_x, center_y, end_x-center_x, end_y-center_y, 
             head_width=arrow_length*0.3, head_length=arrow_length*0.3, 
             fc='black', ec='black', zorder=3)

def visualize_layout(layout, layout_num):
    fig, ax = plt.subplots(figsize=(8, 9))

    # Draw room
    room = patches.Rectangle((0, 0), ROOM_WIDTH, ROOM_HEIGHT,
                             linewidth=2, edgecolor='black', facecolor='none')
    ax.add_patch(room)

    # Define fixture images
    fixture_images = {
        'toilet': "/media/patrick/Patrick/Singularity_AI_Design_Tool/Data/Assets/2d_Images/New Assets/Toilet.png",
        'sink': "/media/patrick/Patrick/Singularity_AI_Design_Tool/Data/Assets/2d_Images/New Assets/sink.png",
        'bathtub': "/media/patrick/Patrick/Singularity_AI_Design_Tool/Data/Assets/2d_Images/New Assets/Tub.png"
    }
    # Ensure door clearance and proper placement
    door_wall = random.choice(['left', 'right', 'top', 'bottom'])
    door_clearance = DOOR_WIDTH  # Minimum clearance from fixtures

    if door_wall == 'left':
        door_x, door_y = -WALL_THICKNESS, random.uniform(door_clearance, ROOM_HEIGHT - DOOR_WIDTH - door_clearance)
        door_width, door_height = WALL_THICKNESS, DOOR_WIDTH
    elif door_wall == 'right':
        door_x, door_y = ROOM_WIDTH, random.uniform(door_clearance, ROOM_HEIGHT - DOOR_WIDTH - door_clearance)
        door_width, door_height = WALL_THICKNESS, DOOR_WIDTH
    elif door_wall == 'top':
        door_x, door_y = random.uniform(door_clearance, ROOM_WIDTH - DOOR_WIDTH - door_clearance), ROOM_HEIGHT
        door_width, door_height = DOOR_WIDTH, WALL_THICKNESS
    else:  # bottom
        door_x, door_y = random.uniform(door_clearance, ROOM_WIDTH - DOOR_WIDTH - door_clearance), -WALL_THICKNESS
        door_width, door_height = DOOR_WIDTH, WALL_THICKNESS

    door = patches.Rectangle((door_x, door_y), door_width, door_height,
                             linewidth=1, edgecolor='brown', facecolor='brown', label='Door')
    # ax.add_patch(door)

    # Draw window (randomly placed on a wall, avoiding door wall)
    window_wall = random.choice([w for w in ['left', 'right', 'top', 'bottom'] if w != door_wall])
    if window_wall == 'left':
        window_x, window_y = -WALL_THICKNESS, random.uniform(0, ROOM_HEIGHT - WINDOW_WIDTH)
        window_width, window_height = WALL_THICKNESS, WINDOW_WIDTH
    elif window_wall == 'right':
        window_x, window_y = ROOM_WIDTH, random.uniform(0, ROOM_HEIGHT - WINDOW_WIDTH)
        window_width, window_height = WALL_THICKNESS, WINDOW_WIDTH
    elif window_wall == 'top':
        window_x, window_y = random.uniform(0, ROOM_WIDTH - WINDOW_WIDTH), ROOM_HEIGHT
        window_width, window_height = WINDOW_WIDTH, WALL_THICKNESS
    else:  # bottom
        window_x, window_y = random.uniform(0, ROOM_WIDTH - WINDOW_WIDTH), -WALL_THICKNESS
        window_width, window_height = WINDOW_WIDTH, WALL_THICKNESS

    window = patches.Rectangle((window_x, window_y), window_width, window_height,
                               linewidth=1, edgecolor='blue', facecolor='blue', label='Window')
    # ax.add_patch(window)

    # Draw fixtures using images
    for fixture_name, image_path in fixture_images.items():
        if fixture_name in layout:
            x, y, z, width, height, angle = layout[fixture_name]
            print(f"x_{fixture_name}={x}, y_{fixture_name}={y}")
            try:
                img = mpimg.imread(image_path)
                ax.imshow(img, extent=(x+width, x, y, y + height), aspect='auto', alpha=1.0)
            except FileNotFoundError:
                print(f"Error: Image not found for {fixture_name} at {image_path}")

    # Set limits and title
    ax.set_xlim(-WALL_THICKNESS, ROOM_WIDTH + WALL_THICKNESS)
    ax.set_ylim(-WALL_THICKNESS, ROOM_HEIGHT + WALL_THICKNESS)
    ax.set_aspect('equal')
    ax.set_title(f'Bathroom Layout #{layout_num + 1}')
    ax.set_xlabel('Width (inches)')
    ax.set_ylabel('Height (inches)')

    plt.tight_layout()

    # Save the output
    output_directory = os.path.join(os.getcwd(), "Generated_Layouts_Wednesday_12_03_2D_new_polygons")
    os.makedirs(output_directory, exist_ok=True)
    output_file = os.path.join(output_directory, f'bathroom_layout_{layout_num + 1}.png')

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
    max_attempts = num_layouts * 10  # Limit total attempts to avoid infinite loops
    
    print(f"Generating {num_layouts} unique bathroom layouts...")
    
    while len(unique_layouts) < num_layouts and attempts < max_attempts:
        layout = generate_valid_layout()
        attempts += 1
        
        if layout is not None:
            # Check if this layout is sufficiently different from existing ones
            is_unique = True
            for existing_layout in unique_layouts:
                similarity_score = 0
                
                for fixture in ['bathtub', 'toilet', 'sink']:
                    ex_x, ex_y = existing_layout[fixture][0], existing_layout[fixture][1]
                    new_x, new_y = layout[fixture][0], layout[fixture][1]
                    
                    # Calculate distance between fixtures in the two layouts
                    distance = math.sqrt((ex_x - new_x)**2 + (ex_y - new_y)**2)
                    
                    # If fixtures are very close and have same orientation, consider them similar
                    if distance < 20 and existing_layout[fixture][5] == layout[fixture][5]:
                        similarity_score += 1
                
                # If all three fixtures are similar, layouts are not unique enough
                if similarity_score >= 4:
                    is_unique = False
                    break
            
            if is_unique:
                unique_layouts.append(layout)
                print(f"Layout {len(unique_layouts)}/{num_layouts} generated")
    
    print(f"Generated {len(unique_layouts)} unique layouts after {attempts} attempts")
    
    # Visualize all unique layouts
    for i, layout in enumerate(unique_layouts):
        visualize_layout(layout, i)
    
    return unique_layouts

# Main execution
if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Generate layouts
    layouts = generate_layouts(NUM_LAYOUTS)
    
    print(f"Successfully generated {len(layouts)} bathroom layouts")
    print("Layout images have been saved as bathroom_layout_X.png files")

    # Print the location of the saved files
    current_dir = os.getcwd()
    output_dir = os.path.join(current_dir, "Generated_Layouts_2D_new_polygons")
    print(f"\nAll layouts should be saved to: {output_dir}")
    
    # Check if files were actually created
    try:
        files_created = [f for f in os.listdir(output_dir) if f.startswith('bathroom_layout_') and f.endswith('.png')]
        print(f"Number of files created: {len(files_created)}")
        
        if len(files_created) == 0:
            print("WARNING: No files were created in the output directory!")
            
            # Check if they were saved to current directory instead
            fallback_files = [f for f in os.listdir(current_dir) if f.startswith('bathroom_layout_') and f.endswith('.png')]
            if len(fallback_files) > 0:
                print(f"Found {len(fallback_files)} files in current directory instead.")
    except Exception as e:
        print(f"Error checking output directory: {e}")
