import matplotlib
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scipy.spatial import Delaunay
import plotly.graph_objects as go
import pyvista as pv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import numpy as np
import collada
matplotlib.use("TkAgg")  # Use non-interactive backend



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
            output_path = os.path.abspath(f"H:/Shared drives/AI Design Tool/00-PG_folder/03-Furniture AI Model/2nd Approach/Bathroom/Solver/2D_Ouput_Samples/bathroom_layout_{index}.png")
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
        door_thickness = 0.5
        door_color = 'black'
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
            zoom_factor = min(width/img.shape[1], depth/img.shape[0]) * 5 if fixture == 'toilet' else min(width/img.shape[1], depth/img.shape[0]) * 5
            img_rotated = np.rot90(img, k=rotation//90) if rotation != 0 else img

            # Create offset image
            imagebox = OffsetImage(img_rotated, zoom=zoom_factor)
            ab = AnnotationBbox(imagebox, (x , y ),
                              frameon=False, box_alignment=(0, 0))
            ax.add_artist(ab)

            # Add label (optional)
            label_x = x + width/2
            label_y = y + depth/2
            ax.text(label_x, label_y, fixture.capitalize(),
                   color='black', ha='center', va='center', fontsize=8)

    def render_3D_plotly(self, solution):
        """Render an interactive 3D plot using Plotly."""
        fig = go.Figure()

        # Define room boundaries
        fig.update_layout(
            scene=dict(
                xaxis_title="Width (inches)",
                yaxis_title="Length (inches)",
                zaxis_title="Height (inches)",
                xaxis=dict(range=[0, self.width]),
                yaxis=dict(range=[0, self.length]),
                zaxis=dict(range=[0, 100])
            ),
            title="Interactive 3D Bathroom Layout"
        )

        # Load and draw fixtures
        fixture_models = {
            'toilet': "H:/Shared drives/AI Design Tool/00-PG_folder/03-Furniture AI Model/2nd Approach/Bathroom/Solver/Assets/3d_Models/toilet.dae",
            'sink': "H:/Shared drives/AI Design Tool/00-PG_folder/03-Furniture AI Model/2nd Approach/Bathroom/Solver/Assets/3d_Models/sink.dae",
            'bathtub': "H:/Shared drives/AI Design Tool/00-PG_folder/03-Furniture AI Model/2nd Approach/Bathroom/Solver/Assets/3d_Models/bathtub.dae"
        }
        for fixture, model_path in fixture_models.items():
            if fixture in solution.keys():
                x, y = solution[fixture]['x'], solution[fixture]['y']
                rotation = solution[fixture]['rotation']

                model = collada.Collada(model_path)
                mesh = model.geometries[0].primitives[0]
                verts = np.array(mesh.vertex)
                scale_factor = 10  # Adjust this value based on your needs
                verts[:, 0] = (verts[:, 0] * scale_factor) + x
                verts[:, 1] = (verts[:, 1] * scale_factor) + y
                verts[:, 2] = verts[:, 2] * scale_factor  # Scaling height properly
                # Add as a 3D mesh
                fig.add_trace(go.Mesh3d(
                    x=verts[:, 0]+x, y=verts[:, 1]+y, z=verts[:, 2],
                    color='gray', opacity=0.5
                ))

        fig.show()
    def render_3D(self, solution, index):
        """Render the 3D bathroom layout using COLLADA models."""
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("3D Bathroom Layout")
        plt.ion()
        # Draw room boundaries
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.length)
        ax.set_zlim(0, 100)  # Assume max height of fixtures
        ax.set_xlabel("Width (inches)")
        ax.set_ylabel("Length (inches)")
        ax.set_zlabel("Height (inches)")

        # Load and draw fixtures
        fixture_models = {
            'toilet': "H:/Shared drives/AI Design Tool/00-PG_folder/03-Furniture AI Model/2nd Approach/Bathroom/Solver/Assets/3d_Models/toilet.dae",
            'sink': "H:/Shared drives/AI Design Tool/00-PG_folder/03-Furniture AI Model/2nd Approach/Bathroom/Solver/Assets/3d_Models/sink.dae",
            'bathtub': "H:/Shared drives/AI Design Tool/00-PG_folder/03-Furniture AI Model/2nd Approach/Bathroom/Solver/Assets/3d_Models/bathtub.dae"
        }

        for fixture, model_path in fixture_models.items():
            if fixture in solution.keys():
                x, y = solution[fixture]['x'], solution[fixture]['y']
                rotation = solution[fixture]['rotation']

                self._draw_3D_fixture(ax, model_path, x, y, rotation)
        plt.show()
        plt.savefig(f"bathroom_layout_{index}.png")



    # def render_3D_pyvista(self, solution):
    #     """Render 3D fixtures interactively with PyVista using real 3D models."""
    #     plotter = pv.Plotter()
    #
    #     # Set up room boundaries
    #     room_bounds = pv.Box(bounds=(0, self.width, 0, self.length, 0, 100))
    #     plotter.add_mesh(room_bounds, color='white', opacity=0.2)
    #
    #     # Load and draw fixtures
    #     fixture_models = {
    #         'toilet': "H:/Shared drives/AI Design Tool/00-PG_folder/03-Furniture AI Model/2nd Approach/Bathroom/Solver/Assets/3d_Models/toilet.dae",
    #         'sink': "H:/Shared drives/AI Design Tool/00-PG_folder/03-Furniture AI Model/2nd Approach/Bathroom/Solver/Assets/3d_Models/sink.dae",
    #         'bathtub': "H:/Shared drives/AI Design Tool/00-PG_folder/03-Furniture AI Model/2nd Approach/Bathroom/Solver/Assets/3d_Models/bathtub.dae"
    #     }
    #
    #     for fixture, model_path in fixture_models.items():
    #         if fixture in solution.keys():
    #             x, y = solution[fixture]['x'], solution[fixture]['y']
    #             rotation = solution[fixture]['rotation']
    #
    #             # Load the 3D model
    #             mesh = pv.read(model_path)  # Load COLLADA model
    #             mesh.translate([x, y, 0])  # Move fixture to position
    #             #mesh.rotate_z(rotation)  # Rotate if needed
    #
    #             # Add model to the scene
    #             plotter.add_mesh(mesh, color="gray", opacity=0.7)
    #     plotter.show(interactive=True)

    def apply_rotation(self,verts, rotation, pivot):
        """Rotates vertices around a pivot point"""
        angle = np.radians(rotation)  # Convert degrees to radians
        cos_a, sin_a = np.cos(angle), np.sin(angle)

        # Shift to pivot point
        verts[:, 0] -= pivot[0]
        verts[:, 1] -= pivot[1]

        # Apply 2D rotation
        rotated_x = verts[:, 0] * cos_a - verts[:, 1] * sin_a
        rotated_y = verts[:, 0] * sin_a + verts[:, 1] * cos_a

        # Shift back
        verts[:, 0] = rotated_x + pivot[0]
        verts[:, 1] = rotated_y + pivot[1]

        return verts
    def _draw_3D_fixture(self, ax, model_path, x, y, rotation):
        """Load and draw a 3D fixture from a COLLADA file."""
        try:
            model = collada.Collada(model_path)
            mesh = model.geometries[0].primitives[0]
            verts = np.array(mesh.vertex)
            # Define pivot point (center of fixture or lower-left corner)
            pivot = np.array([np.mean(verts[:, 0]), np.mean(verts[:, 1])])

            # Apply correct positioning
            verts = self.apply_rotation(verts, rotation, pivot)  # Fix rotation
            verts[:, 0] += x  # Adjust X position
            verts[:, 0] *= 50  # Adjust X position

            verts[:, 1] += y  # Adjust Y position
            verts[:, 1] *= 50  # Adjust Y position

            verts[:, 2] *= 10  # Scale height (optional)

            # Generate smoother triangular patches using Delaunay triangulation
            tri = Delaunay(verts[:, :2])  # Use only X, Y for triangulation
            ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], triangles=tri.simplices, cmap='gray', alpha=0.7)

        except Exception as e:
            print(f"Error loading {model_path}: {e}")


    def clear(self):
        """Clear the current plot."""
        plt.clf()


# Main script to run the program
# if __name__ == "__main__":
#     # Load the dataset
#     X_test = pd.read_csv("X_test.csv")
#     model = XGBRegressor()
#     model.load_model("bathroom_model.json")  # Load trained model
#
#     # Predict fixture positions
#     predictions = model.predict(X_test.iloc[0:1])  # Predict for the first example
#
#     # Convert predictions to dictionary format
#     pred_dict = {
#         'Toilet_X_Position': predictions[0][0], 'Toilet_Y_Position': predictions[0][1],
#         'Toilet_Rotation': int(predictions[0][2]),
#         'Sink_X_Position': predictions[0][3], 'Sink_Y_Position': predictions[0][4],
#         'Sink_Rotation': int(predictions[0][5]),
#         'Bathtub_X_Position': predictions[0][6], 'Bathtub_Y_Position': predictions[0][7],
#         'Bathtub_Rotation': int(predictions[0][8]),
#         'Has_Toilet': 1, 'Has_Sink': 1, 'Has_Bathtub': 1  # Assume all fixtures exist for now
#     }
#
#     # Create visualizer and render layout
#     visualizer = BathroomVisualizer(120, 100, 30, 0, 30)  # Example room dimensions
#     visualizer.render(pred_dict, index=0)  # Render 2D
#     visualizer.render_3D(pred_dict, index=0)  # Render 3D
