import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
import math
import os
from constants import bathroom_dataset

class BathroomPreviewHandler:
    def __init__(self):
        self.dataset = bathroom_dataset
        self.room_size = (120, 96)
        self.room_height = 96
        
        # Get the path to the assets folder
        self.assets_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'assets'
        )
        
        # Load preview images
        self.load_preview_assets()
        
        self.element_positions = {
            'toilet': (30, 24),
            'sink': (60, 24),
            'bathtub': (30, 70)
        }

    def load_preview_assets(self):
        """Load all preview images"""
        self.previews_2d = {}
        self.previews_3d = {}
        
        for element_type in ['toilet', 'sink', 'bathtub']:
            # Load 2D preview
            try:
                path_2d = os.path.join(self.assets_path, '2d_previews', f'{element_type}_top.png')
                if os.path.exists(path_2d):
                    self.previews_2d[element_type] = mpimg.imread(path_2d)
                    print(f"Loaded 2D preview for {element_type}")
                else:
                    print(f"2D preview not found: {path_2d}")
            except Exception as e:
                print(f"Error loading 2D preview for {element_type}: {e}")

            # Load 3D preview
            try:
                path_3d = os.path.join(self.assets_path, '3d_previews', f'{element_type}_iso.png')
                if os.path.exists(path_3d):
                    self.previews_3d[element_type] = mpimg.imread(path_3d)
                    print(f"Loaded 3D preview for {element_type}")
                else:
                    print(f"3D preview not found: {path_3d}")
            except Exception as e:
                print(f"Error loading 3D preview for {element_type}: {e}")

    def _draw_2d_element(self, ax, element_type, position):
        """Draw 2D element using preview image"""
        x, y = position
        dims = self.dataset["elements_rules"][element_type]["dimensions"]
        width = sum(dims["width_range"]) / 2
        depth = sum(dims["depth_range"]) / 2

        if element_type in self.previews_2d:
            # Create OffsetImage with preview
            preview = self.previews_2d[element_type]
            im = OffsetImage(preview, zoom=width/100)  # Adjust zoom factor as needed
            ab = AnnotationBbox(
                im, 
                (x, y),
                frameon=False,
                pad=0
            )
            ax.add_artist(ab)
        else:
            # Fallback to rectangle if preview not available
            rect = patches.Rectangle(
                (x - width/2, y - depth/2),
                width, depth,
                color='gray',
                alpha=0.5
            )
            ax.add_patch(rect)

        # Draw clearance zone
        clearance = self.dataset["elements_rules"][element_type]["placement"]
        self._draw_clearance_zone(ax, position, clearance)

    def _draw_3d_element(self, ax, element_type, position):
        """Draw 3D element using preview image"""
        x, y = position
        dims = self.dataset["elements_rules"][element_type]["dimensions"]
        height = sum(dims["height_range"]) / 2

        if element_type in self.previews_3d:
            # Create a vertical image plane for 3D preview
            preview = self.previews_3d[element_type]
            
            # Calculate image dimensions while maintaining aspect ratio
            img_height = height
            img_width = img_height * preview.shape[1] / preview.shape[0]
            
            # Create image plane vertices
            vertices = np.array([
                [x - img_width/2, y, 0],
                [x + img_width/2, y, 0],
                [x + img_width/2, y, img_height],
                [x - img_width/2, y, img_height]
            ])
            
            # Plot the image on a 3D plane
            ax.plot_surface(
                vertices[:, 0].reshape(2, 2),
                vertices[:, 1].reshape(2, 2),
                vertices[:, 2].reshape(2, 2),
                facecolors=preview,
                shade=False
            )
        else:
            # Fallback to simple 3D box
            width = sum(dims["width_range"]) / 2
            depth = sum(dims["depth_range"]) / 2
            
            # Draw simple box
            ax.bar3d(
                x - width/2, y - depth/2, 0,
                width, depth, height,
                color='gray',
                alpha=0.5
            )

    def show_preview(self):
        """Show combined preview of bathroom elements"""
        fig = plt.figure(figsize=(15, 10))
        
        # 2D Floor Plan
        ax1 = fig.add_subplot(221)
        self._draw_2d_plan(ax1, "Floor Plan")
        
        # 2D Ceiling Plan
        ax2 = fig.add_subplot(222)
        self._draw_2d_plan(ax2, "Ceiling Plan")
        
        # 3D View
        ax3 = fig.add_subplot(223, projection='3d')
        self._draw_3d_view(ax3)
        
        # Dimensions
        ax4 = fig.add_subplot(224)
        self._draw_dimensions(ax4)
        
        plt.tight_layout()
        plt.show()

    def _draw_2d_plan(self, ax, title):
        """Draw 2D plan with elements"""
        # Draw room boundary
        ax.plot([0, self.room_size[0], self.room_size[0], 0, 0],
                [0, 0, self.room_size[1], self.room_size[1], 0],
                'k-', linewidth=2)

        # Draw elements
        for element_type, position in self.element_positions.items():
            self._draw_2d_element(ax, element_type, position)

        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_title(title)
        ax.set_xlabel('Width (inches)')
        ax.set_ylabel('Length (inches)')
        ax.set_xlim(-10, self.room_size[0] + 10)
        ax.set_ylim(-10, self.room_size[1] + 10)

    def _draw_3d_view(self, ax):
        """Draw 3D view of bathroom"""
        # Draw room boundaries
        ax.plot([0, self.room_size[0], self.room_size[0], 0, 0],
                [0, 0, self.room_size[1], self.room_size[1], 0],
                [0, 0, 0, 0, 0], 'k-', linewidth=2)
        
        # Draw walls
        for x, y in [(0,0), (self.room_size[0],0), 
                     (self.room_size[0],self.room_size[1]), (0,self.room_size[1])]:
            ax.plot([x,x], [y,y], [0,self.room_height], 'k-', alpha=0.3)

        # Draw elements
        for element_type, position in self.element_positions.items():
            self._draw_3d_element(ax, element_type, position)

        ax.view_init(elev=30, azim=45)
        ax.set_title('3D View')
        ax.set_xlabel('Width (inches)')
        ax.set_ylabel('Length (inches)')
        ax.set_zlabel('Height (inches)')

    def _draw_clearance_zone(self, ax, position, clearance):
        """Draw clearance zone for element"""
        x, y = position
        front = clearance["clearance_front"]
        sides = clearance["clearance_sides"]

        rect = patches.Rectangle(
            (x - sides, y - front/2),
            sides * 2,
            front,
            color='gray',
            alpha=0.2,
            linestyle='--',
            fill=False
        )
        ax.add_patch(rect)

    def _draw_dimensions(self, ax):
        """Draw dimension information"""
        info = [
            "Bathroom Dimensions",
            f"Room: {self.room_size[0]}\" × {self.room_size[1]}\" × {self.room_height}\"",
            f"Area: {self.room_size[0] * self.room_size[1]} sq.in",
            "\nElement Dimensions (min-max):"
        ]
        
        for element_type, rules in self.dataset["elements_rules"].items():
            dims = rules["dimensions"]
            info.append(
                f"\n{element_type.title()}:\n"
                f"  Width: {dims['width_range'][0]}\"-{dims['width_range'][1]}\"\n"
                f"  Depth: {dims['depth_range'][0]}\"-{dims['depth_range'][1]}\"\n"
                f"  Height: {dims['height_range'][0]}\"-{dims['height_range'][1]}\""
            )
        
        ax.text(0.05, 0.95, '\n'.join(info),
                transform=ax.transAxes,
                verticalalignment='top',
                fontfamily='monospace')
        ax.axis('off')

def main():
    """Test the bathroom preview handler"""
    preview = BathroomPreviewHandler()
    preview.show_preview()

if __name__ == "__main__":
    main()
