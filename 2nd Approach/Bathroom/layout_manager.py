from bathroom_element import BathroomElement
from shapely.geometry import Polygon, Point, LineString
import math

class BathroomLayout:
    def __init__(self, room_boundary):
        """
        Initialize bathroom layout
        
        Args:
            room_boundary (list): List of (x,y) tuples defining room corners in inches
        """
        try:
            # Store original boundary points
            self.boundary = room_boundary
            # Initialize elements list
            self.elements = []
            # Create Shapely polygon from boundary points
            self.boundary_polygon = Polygon(self.boundary)
            
            # Validate room boundary
            if not self.boundary_polygon.is_valid:
                raise ValueError("Invalid room boundary provided")
                
            print("Room boundary initialized successfully")
            print(f"Room area: {self.boundary_polygon.area:.2f} square inches")
            
        except Exception as e:
            print(f"Error initializing room boundary: {str(e)}")
            # Provide default rectangular room if initialization fails
            self.boundary = [(0,0), (120,0), (120,96), (0,96)]
            self.boundary_polygon = Polygon(self.boundary)
            print("Falling back to default room dimensions")

    def validate_element_placement(self, element):
        """Validate element placement against all rules"""
        try:
            boundary_check = self._check_boundary_collision(element)
            if not boundary_check:
                print(f"Failed boundary collision check for {element.type}")
                return False
                
            element_check = self._check_element_collisions(element)
            if not element_check:
                print(f"Failed element collision check for {element.type}")
                return False
                
            return True
        except Exception as e:
            print(f"Error validating element placement: {str(e)}")
            return False

    def _check_boundary_collision(self, element):
        """Check if element collides with room boundary"""
        try:
            # Get element corners
            element_corners = element.get_element_corners()
            element_polygon = Polygon(element_corners)
            
            # Check if element is completely inside boundary
            if not self.boundary_polygon.contains(element_polygon):
                print(f"Element {element.type} outside room boundaries")
                return False
            
            # Get clearance requirements
            clearance = element.get_clearance_zone()
            clearance_points = self._generate_clearance_polygon(
                element.position,
                clearance["front"],
                clearance["sides"],
                element.rotation
            )
            
            # Create clearance polygon
            clearance_polygon = Polygon(clearance_points)
            
            # Check if clearance zone is within boundary
            if not self.boundary_polygon.contains(clearance_polygon):
                print(f"Clearance zone for {element.type} outside room boundaries")
                return False
                
            return True
            
        except Exception as e:
            print(f"Error checking boundary collision: {str(e)}")
            return False

    def _check_element_collisions(self, new_element):
        """Check if new element collides with existing elements"""
        try:
            new_corners = new_element.get_element_corners()
            new_polygon = Polygon(new_corners)
            
            for existing_element in self.elements:
                existing_corners = existing_element.get_element_corners()
                existing_polygon = Polygon(existing_corners)
                
                # Check physical overlap
                if new_polygon.intersects(existing_polygon):
                    print(f"Collision detected between {new_element.type} and {existing_element.type}")
                    return False
                
                # Check clearance zones
                new_clearance = new_element.get_clearance_zone()
                existing_clearance = existing_element.get_clearance_zone()
                
                new_clearance_poly = Polygon(self._generate_clearance_polygon(
                    new_element.position,
                    new_clearance["front"],
                    new_clearance["sides"],
                    new_element.rotation
                ))
                
                existing_clearance_poly = Polygon(self._generate_clearance_polygon(
                    existing_element.position,
                    existing_clearance["front"],
                    existing_clearance["sides"],
                    existing_element.rotation
                ))
                
                if new_clearance_poly.intersects(existing_clearance_poly):
                    print(f"Clearance zone overlap between {new_element.type} and {existing_element.type}")
                    return False
                    
            return True
            
        except Exception as e:
            print(f"Error checking element collisions: {str(e)}")
            return False

    def _generate_clearance_polygon(self, position, front_clearance, side_clearance, rotation):
        """Generate polygon points for clearance zone"""
        x, y = position
        theta = math.radians(rotation)
        
        points = [
            # Front right
            (x + front_clearance * math.sin(theta) + side_clearance * math.cos(theta),
             y + front_clearance * math.cos(theta) - side_clearance * math.sin(theta)),
            # Front left
            (x + front_clearance * math.sin(theta) - side_clearance * math.cos(theta),
             y + front_clearance * math.cos(theta) + side_clearance * math.sin(theta)),
            # Back left
            (x - front_clearance * math.sin(theta) - side_clearance * math.cos(theta),
             y - front_clearance * math.cos(theta) + side_clearance * math.sin(theta)),
            # Back right
            (x - front_clearance * math.sin(theta) + side_clearance * math.cos(theta),
             y - front_clearance * math.cos(theta) - side_clearance * math.sin(theta))
        ]
        
        return points

    def add_element(self, element):
        """Add element to layout if valid"""
        try:
            if self.validate_element_placement(element):
                self.elements.append(element)
                print(f"Successfully added {element.type} at position {element.position}")
                return True
            return False
        except Exception as e:
            print(f"Error adding element: {str(e)}")
            return False

if __name__ == "__main__":
    # Test the layout manager
    def test_layout():
        print("Starting layout tests...")
        
        # Create a rectangular room (120x96 inches)
        room_boundary = [(0,0), (120,0), (120,96), (0,96)]
        layout = BathroomLayout(room_boundary)
        
        # Test cases
        test_cases = [
            # Test case 1: Valid placement
            ("toilet", (30, 24), 0),
            # Test case 2: Too close to wall
            ("toilet", (5, 5), 0),
            # Test case 3: Valid sink placement
            ("sink", (60, 24), 0),
            # Test case 4: Overlapping elements
            ("sink", (30, 24), 0)
        ]
        
        # Run tests
        for i, (element_type, position, rotation) in enumerate(test_cases, 1):
            print(f"\nTest case {i}:")
            print(f"Attempting to place {element_type} at position {position}")
            
            element = BathroomElement(element_type, position, rotation)
            result = layout.add_element(element)
            
            print(f"Result: {'Success' if result else 'Failed'}")
            print(f"Current elements in layout: {len(layout.elements)}")
            
        print("\nTest complete")

    test_layout()
