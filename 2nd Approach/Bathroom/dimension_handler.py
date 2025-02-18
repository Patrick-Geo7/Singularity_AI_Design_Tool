import math
from shapely.geometry import Polygon

class DimensionHandler:
    def __init__(self, layout):
        """
        Initialize dimension handler
        
        Args:
            layout (BathroomLayout): Layout to generate dimensions for
        """
        self.layout = layout
        
    def generate_all_dimensions(self):
        """
        Generate comprehensive dimension data for the entire layout
        
        Returns:
            dict: Complete dimension data including all measurements
        """
        return {
            'room': self.get_room_dimensions(),
            'elements': self.get_element_dimensions(),
            'clearances': self.get_clearance_dimensions(),
            'relationships': self.get_relationship_dimensions(),
            'metadata': self.get_dimension_metadata()
        }
    
    def get_room_dimensions(self):
        """Calculate room dimensions and area"""
        boundary = self.layout.boundary
        room_dims = {
            'width': abs(max(x for x, _ in boundary) - min(x for x, _ in boundary)),
            'length': abs(max(y for _, y in boundary) - min(y for _, y in boundary)),
            'area': Polygon(boundary).area,
            'perimeter': self._calculate_perimeter(boundary)
        }
        return room_dims
    
    def get_element_dimensions(self):
        """Get detailed dimensions for all elements"""
        element_dims = {}
        for element in self.layout.elements:
            corners = element.get_element_corners()
            polygon = Polygon(corners)
            
            element_dims[element.type] = {
                'width': abs(max(x for x, _ in corners) - min(x for x, _ in corners)),
                'depth': abs(max(y for _, y in corners) - min(y for _, y in corners)),
                'height': element.rules['dimensions']['height_range'][1],  # Using max height
                'area': polygon.area,
                'position': element.position,
                'rotation': element.rotation,
                'clearance_required': {
                    'front': element.rules['placement']['clearance_front'],
                    'sides': element.rules['placement']['clearance_sides']
                }
            }
        return element_dims
    
    def get_clearance_dimensions(self):
        """Calculate clearance zones and areas"""
        clearance_dims = {}
        for element in self.layout.elements:
            clearance = element.get_clearance_zone()
            points = self.layout._generate_clearance_polygon(
                element.position,
                clearance["front"],
                clearance["sides"],
                element.rotation
            )
            
            clearance_dims[element.type] = {
                'front': clearance["front"],
                'sides': clearance["sides"],
                'area': Polygon(points).area,
                'points': points,
                'usable_space': self._calculate_usable_space(points)
            }
        return clearance_dims
    
    def get_relationship_dimensions(self):
        """Calculate distances and relationships between elements"""
        relationships = {}
        elements = self.layout.elements
        
        for i in range(len(elements)):
            for j in range(i + 1, len(elements)):
                elem1, elem2 = elements[i], elements[j]
                distance = math.sqrt(
                    (elem1.position[0] - elem2.position[0])**2 +
                    (elem1.position[1] - elem2.position[1])**2
                )
                
                key = f"{elem1.type}_to_{elem2.type}"
                relationships[key] = {
                    'direct_distance': distance,
                    'walking_path': self._calculate_walking_path(elem1, elem2),
                    'clearance_overlap': self._check_clearance_overlap(elem1, elem2),
                    'relative_positions': {
                        'element1': elem1.position,
                        'element2': elem2.position,
                        'angle': self._calculate_angle(elem1, elem2)
                    }
                }
        return relationships
    
    def get_dimension_metadata(self):
        """Generate metadata about the dimensions"""
        return {
            'units': 'inches',
            'total_elements': len(self.layout.elements),
            'room_efficiency': self._calculate_room_efficiency(),
            'timestamp': '2024-12-24 14:46:37'
        }
    
    def _calculate_perimeter(self, points):
        """Calculate perimeter of a polygon"""
        perimeter = 0
        for i in range(len(points)):
            p1 = points[i]
            p2 = points[(i + 1) % len(points)]
            perimeter += math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        return perimeter
    
    def _calculate_usable_space(self, points):
        """Calculate usable space within clearance zone"""
        return Polygon(points).area
    
    def _calculate_walking_path(self, elem1, elem2):
        """Calculate walking path distance between elements"""
        # Simplified walking path calculation
        direct_distance = math.sqrt(
            (elem1.position[0] - elem2.position[0])**2 +
            (elem1.position[1] - elem2.position[1])**2
        )
        return direct_distance * 1.2  # Adding 20% for realistic walking path
    
    def _check_clearance_overlap(self, elem1, elem2):
        """Check if clearance zones overlap"""
        clearance1 = Polygon(self.layout._generate_clearance_polygon(
            elem1.position,
            elem1.get_clearance_zone()["front"],
            elem1.get_clearance_zone()["sides"],
            elem1.rotation
        ))
        clearance2 = Polygon(self.layout._generate_clearance_polygon(
            elem2.position,
            elem2.get_clearance_zone()["front"],
            elem2.get_clearance_zone()["sides"],
            elem2.rotation
        ))
        return clearance1.intersects(clearance2)
    
    def _calculate_angle(self, elem1, elem2):
        """Calculate angle between elements"""
        return math.degrees(math.atan2(
            elem2.position[1] - elem1.position[1],
            elem2.position[0] - elem1.position[0]
        ))
    
    def _calculate_room_efficiency(self):
        """Calculate room space efficiency"""
        room_area = Polygon(self.layout.boundary).area
        used_area = sum(Polygon(elem.get_element_corners()).area 
                       for elem in self.layout.elements)
        return (used_area / room_area) * 100 if room_area > 0 else 0
