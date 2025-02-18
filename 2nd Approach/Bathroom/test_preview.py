import matplotlib.pyplot as plt
from layout_manager import BathroomLayout
from bathroom_element import BathroomElement
from preview_handler import PreviewHandler

def test_basic_preview():
    """
    Basic test to verify preview functionality
    """
    try:
        # 1. Create a simple room
        print("Creating room layout...")
        room_boundary = [(0,0), (120,0), (120,96), (0,96)]
        layout = BathroomLayout(room_boundary)
        
        # 2. Add a single element
        print("Adding test element...")
        toilet = BathroomElement("toilet", (30, 24), 0)
        layout.add_element(toilet)
        
        # 3. Create preview handler
        print("Creating preview handler...")
        preview = PreviewHandler(layout)
        
        # 4. Test simple 2D preview first
        print("Generating 2D preview...")
        fig, ax = plt.subplots(figsize=(10, 8))
        preview._draw_2d_preview(ax, "floor_plan")
        plt.show()
        
        print("Preview test completed successfully")
        
    except Exception as e:
        print(f"Error during preview test: {str(e)}")
        raise

if __name__ == "__main__":
    print("Starting preview test...")
    test_basic_preview()
