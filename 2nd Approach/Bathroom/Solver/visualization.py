import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import os
import numpy as np

class BathroomVisualizer:
    def __init__(self, width, length, door_x, door_y, door_width, fixture_images=None):
        """Initialize visualizer with room dimensions and fixture images."""
        self.width = width
        self.length = length
        self.door_x = door_x
        self.door_y = door_y
        self.door_width = door_width
        self.fixture_images = fixture_images or {
            'toilet': 'toilet.png',
            'sink': 'sink.png',
            'bathtub': 'bathtub.png'
        }
        self._loaded_images = {}

    def render(self, solution, index):
        """Render the bathroom layout using matplotlib."""
        plt.clf()
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_xlim(-5, self.width + 5)
        ax.set_ylim(-5, self.length + 5)
        ax.set_title("Bathroom Layout")

        # Draw room outline
        ax.add_patch(patches.Rectangle((0, 0), self.width, self.length,
                                   edgecolor='black', facecolor='none', lw=2))

        # Draw door if provided
        if all(x is not None for x in [self.door_x, self.door_y, self.door_width]):
            self._draw_door(ax, self.door_x, self.door_y, self.door_width)

        # Draw fixtures with images
        self._draw_fixtures(ax, solution)

        plt.xlabel("Width (inches)")
        plt.ylabel("Length (inches)")
        plt.grid(True, linestyle='--', alpha=0.6)

        try:
            output_path = os.path.abspath(f"bathroom_layout_{index}.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Error saving visualization: {str(e)}")
            raise

    def _load_image(self, fixture_type):
        """Load and cache fixture images."""
        if fixture_type not in self._loaded_images:
            path = self.fixture_images[fixture_type]
            img = plt.imread(path)
            self._loaded_images[fixture_type] = img
        return self._loaded_images[fixture_type]

    def _draw_door(self, ax, door_x, door_y, door_width):
        """Draw door and swing arc."""
        door_thickness = 2
        door_color = 'blue'
        door_alpha = 0.5

        # Draw door based on wall placement
        if door_y == 0:  # Bottom wall
            ax.add_patch(patches.Rectangle((door_x, 0), door_width, door_thickness,
                                      edgecolor=door_color, facecolor=door_color, alpha=door_alpha))
            door_center = (door_x + door_width/2, 0)
            door_angle = 0
        elif door_x == 0:  # Left wall
            ax.add_patch(patches.Rectangle((0, door_y), door_thickness, door_width,
                                      edgecolor=door_color, facecolor=door_color, alpha=door_alpha))
            door_center = (0, door_y + door_width/2)
            door_angle = 90
        elif door_x == self.width:  # Right wall
            ax.add_patch(patches.Rectangle((self.width - door_thickness, door_y),
                                      door_thickness, door_width,
                                      edgecolor=door_color, facecolor=door_color, alpha=door_alpha))
            door_center = (self.width, door_y + door_width/2)
            door_angle = 270
        else:  # Top wall
            ax.add_patch(patches.Rectangle((door_x, self.length - door_thickness),
                                      door_width, door_thickness,
                                      edgecolor=door_color, facecolor=door_color, alpha=door_alpha))
            door_center = (door_x + door_width/2, self.length)
            door_angle = 180

        # Add door swing arc
        swing_radius = door_width
        arc = patches.Arc(door_center, swing_radius, swing_radius,
                         theta1=door_angle, theta2=door_angle+90,
                         edgecolor=door_color, linestyle='--', alpha=0.3)
        ax.add_patch(arc)

    def _draw_fixtures(self, ax, solution):
        """Draw bathroom fixtures using PNG images."""
        dimensions = {
            'toilet': (19, 28),
            'sink': (30, 20),
            'bathtub': (30, 60)
        }

        for fixture, pos in solution.items():
            x, y = pos['x'], pos['y']
            rotation = pos['rotation']
            width, depth = dimensions[fixture]

            # Adjust dimensions based on rotation
            if rotation in [90, 270]:
                width, depth = depth, width

            # Load and rotate image
            img = self._load_image(fixture)
            zoom_factor = min(width/img.shape[1], depth/img.shape[0]) * 7.5 if fixture == 'toilet' else min(width/img.shape[1], depth/img.shape[0]) * 5
            img_rotated = np.rot90(img, k=rotation//90) if rotation != 0 else img

            # Create offset image
            imagebox = OffsetImage(img_rotated, zoom=zoom_factor)
            ab = AnnotationBbox(imagebox, (x + width/2, y + depth/2),
                              frameon=False, box_alignment=(0.5, 0.5))
            ax.add_artist(ab)

            # Add label (optional)
            label_x = x + width/2
            label_y = y + depth/2
            ax.text(label_x, label_y, fixture.capitalize(),
                   color='black', ha='center', va='center', fontsize=8)

    # Keep other methods (_draw_door, _add_direction_indicator, clear) the same