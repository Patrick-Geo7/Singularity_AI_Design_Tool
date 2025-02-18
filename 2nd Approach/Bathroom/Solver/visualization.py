import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

class BathroomVisualizer:
    def __init__(self, width, length):
        """Initialize visualizer with room dimensions."""
        self.width = width
        self.length = length

    def render(self, solution,index ,door_x=None, door_y=None, door_width=None):
        """Render the bathroom layout using matplotlib."""
        # Clear any existing plots
        plt.clf()

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_xlim(-5, self.width + 5)
        ax.set_ylim(-5, self.length + 5)
        ax.set_title("Bathroom Layout")

        # Draw room outline
        ax.add_patch(patches.Rectangle((0, 0), self.width, self.length,
                                   edgecolor='black', facecolor='none', lw=2))

        # Draw door if provided
        if all(x is not None for x in [door_x, door_y, door_width]):
            self._draw_door(ax, door_x, door_y, door_width)

        # Draw fixtures
        self._draw_fixtures(ax, solution)

        plt.xlabel("Width (inches)")
        plt.ylabel("Length (inches)")
        plt.grid(True, linestyle='--', alpha=0.6)

        # Save the plot
        try:
            output_path = os.path.abspath(f"'bathroom_layout_{index}'.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Error saving visualization: {str(e)}")
            raise

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
        arc = patches.Arc(door_center, swing_radius*2, swing_radius*2,
                         theta1=door_angle, theta2=door_angle+90,
                         edgecolor=door_color, linestyle='--', alpha=0.3)
        ax.add_patch(arc)

    def _draw_fixtures(self, ax, solution):
        """Draw bathroom fixtures with proper dimensions and orientation."""
        dimensions = {
            'toilet': (19, 28),  # Width x Depth
            'sink': (30, 20),    # Width x Depth
            'bathtub': (30, 60)  # Width x Depth
        }
        colors = {
            'toilet': 'red',
            'sink': 'green',
            'bathtub': 'purple'
        }

        for fixture, pos in solution.items():
            x, y = pos['x'], pos['y']
            rotation = pos['rotation']
            width, depth = dimensions[fixture]

            # Adjust dimensions based on rotation (90° or 270° rotations swap width/depth)
            if rotation in [90, 270]:
                width, depth = depth, width

            # Create fixture rectangle
            rect = patches.Rectangle((x, y), width, depth,
                                 edgecolor=colors[fixture],
                                 facecolor=colors[fixture],
                                 alpha=0.5)
            ax.add_patch(rect)

            # Add fixture label
            label_x = x + width/2
            label_y = y + depth/2
            ax.text(label_x, label_y, fixture.capitalize(),
                   color='white', ha='center', va='center', fontsize=10)

            # Add direction indicator
            self._add_direction_indicator(ax, x, y, width, depth, rotation, colors[fixture])

    def _add_direction_indicator(self, ax, x, y, width, depth, rotation, color):
        """Add an arrow showing which way the fixture is facing."""
        arrow_length = min(width, depth) * 0.2
        arrow_width = arrow_length * 0.3
        arrow_alpha = 0.7

        # Calculate arrow position based on rotation
        if rotation == 0:  # Facing up
            arrow_x = x + width/2
            arrow_y = y + depth
            dx, dy = 0, arrow_length
        elif rotation == 90:  # Facing right
            arrow_x = x + width
            arrow_y = y + depth/2
            dx, dy = arrow_length, 0
        elif rotation == 180:  # Facing down
            arrow_x = x + width/2
            arrow_y = y
            dx, dy = 0, -arrow_length
        else:  # rotation == 270, Facing left
            arrow_x = x
            arrow_y = y + depth/2
            dx, dy = -arrow_length, 0

        ax.arrow(arrow_x, arrow_y, dx, dy,
                head_width=arrow_width, head_length=arrow_width,
                color=color, alpha=arrow_alpha)

    def clear(self):
        """Clear the current plot."""
        plt.clf()