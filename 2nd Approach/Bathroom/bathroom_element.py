from constants import bathroom_dataset
import math

class BathroomElement:
    def __init__(self, element_type, position, rotation=0):
        """
        Initialize a bathroom element
        
        Args:
            element_type (str): Type of element (toilet, sink, etc.)
            position (tuple): (x, y) position in inches
            rotation (float): Rotation in degrees
        """
        self.type = element_type
        self.position = position
        self.rotation = rotation
        self.rules = bathroom_dataset["elements_rules"][element_type]
        
    def validate_placement(self, room_context):
        """Validate element placement within room context"""
        clearance = self.get_clearance_zone()
        wall_distance = self._calculate_wall_distance(room_context)
        
        return (wall_distance >= self.rules["placement"]["min_wall_distance"] and
                self._check_clearance(clearance, room_context))
    
    def get_clearance_zone(self):
        """Calculate the clearance zone for the element"""
        front_clearance = self.rules["placement"]["clearance_front"]
        side_clearance = self.rules["placement"].get("clearance_sides", 0)
        
        return {
            "front": front_clearance,
            "sides": side_clearance,
            "position": self.position,
            "rotation": self.rotation
        }
    
    def _calculate_wall_distance(self, room_context):
        """
        Calculate the minimum distance from the element to any wall in the room
        
        Args:
            room_context (BathroomLayout): The layout containing room boundary
            
        Returns:
            float: Minimum distance to nearest wall in inches
        """
        def point_to_line_distance(point, line_start, line_end):
            """Calculate distance from a point to a line segment"""
            x0, y0 = point
            x1, y1 = line_start
            x2, y2 = line_end
            
            # Calculate numerator
            numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
            # Calculate denominator
            denominator = math.sqrt((y2-y1)**2 + (x2-x1)**2)
            
            # Check if point is beyond line segment ends
            if denominator == 0:
                return math.sqrt((x0-x1)**2 + (y0-y1)**2)
                
            # Calculate projection point parameter
            t = ((x0-x1)*(x2-x1) + (y0-y1)*(y2-y1)) / (denominator**2)
            
            if t < 0:  # Point is beyond start of line
                return math.sqrt((x0-x1)**2 + (y0-y1)**2)
            elif t > 1:  # Point is beyond end of line
                return math.sqrt((x0-x2)**2 + (y0-y2)**2)
            
            # Return perpendicular distance
            return numerator / denominator

        # Get element position
        element_x, element_y = self.position
        
        # Get element dimensions
        width = sum(self.rules["dimensions"]["width_range"]) / 2  # Average width
        depth = sum(self.rules["dimensions"]["depth_range"]) / 2  # Average depth
        
        # Calculate element corners based on rotation
        theta = math.radians(self.rotation)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        
        # Element corner points relative to center
        corners = [
            (element_x + (width/2)*cos_t - (depth/2)*sin_t,
             element_y + (width/2)*sin_t + (depth/2)*cos_t),
            (element_x + (width/2)*cos_t + (depth/2)*sin_t,
             element_y + (width/2)*sin_t - (depth/2)*cos_t),
            (element_x - (width/2)*cos_t + (depth/2)*sin_t,
             element_y - (width/2)*sin_t - (depth/2)*cos_t),
            (element_x - (width/2)*cos_t - (depth/2)*sin_t,
             element_y - (width/2)*sin_t + (depth/2)*cos_t)
        ]
        
        # Calculate minimum distance from each corner to each wall
        min_distance = float('inf')
        
        for i in range(len(room_context.boundary)):
            wall_start = room_context.boundary[i]
            wall_end = room_context.boundary[(i + 1) % len(room_context.boundary)]
            
            # Check distance from each corner to current wall
            for corner in corners:
                distance = point_to_line_distance(corner, wall_start, wall_end)
                min_distance = min(min_distance, distance)
        
        return min_distance

    def get_element_corners(self):
        """
        Get the corner points of the element for visualization or debugging
        
        Returns:
            list: List of (x,y) tuples representing element corners
        """
        element_x, element_y = self.position
        width = sum(self.rules["dimensions"]["width_range"]) / 2
        depth = sum(self.rules["dimensions"]["depth_range"]) / 2
        
        theta = math.radians(self.rotation)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        
        return [
            (element_x + (width/2)*cos_t - (depth/2)*sin_t,
             element_y + (width/2)*sin_t + (depth/2)*cos_t),
            (element_x + (width/2)*cos_t + (depth/2)*sin_t,
             element_y + (width/2)*sin_t - (depth/2)*cos_t),
            (element_x - (width/2)*cos_t + (depth/2)*sin_t,
             element_y - (width/2)*sin_t - (depth/2)*cos_t),
            (element_x - (width/2)*cos_t - (depth/2)*sin_t,
             element_y - (width/2)*sin_t + (depth/2)*cos_t)
        ]
    
if __name__ == "__main__":
    # Test the wall distance calculation
    class MockLayout:
        def __init__(self):
            # Example room boundary (rectangular room 120x96 inches)
            self.boundary = [(0,0), (120,0), (120,96), (0,96)]
    
    # Create a test element (toilet) placed at various positions
    test_positions = [
        (24, 24),    # Near corner
        (60, 48),    # Middle of room
        (12, 48),    # Near wall
        (108, 48),   # Near opposite wall
    ]
    
    mock_layout = MockLayout()
    
    for pos in test_positions:
        element = BathroomElement("toilet", pos, rotation=0)
        distance = element._calculate_wall_distance(mock_layout)
        print(f"Element at position {pos}: Distance to nearest wall = {distance:.2f} inches")
        
        # Get and print corner positions for verification
        corners = element.get_element_corners()
        print(f"Element corners: {corners}\n")
