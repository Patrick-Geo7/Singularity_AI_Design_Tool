from constants import element_dimensions, clearances, mandatory_rules

class DimensionRules:
    def __init__(self):
        self.element_dimensions = element_dimensions
        self.clearances = clearances
        self.mandatory_rules = mandatory_rules

    def get_element_dimensions(self, element_type):
        """
        Get dimensions for a specific element type
        Args:
            element_type: string ('toilet', 'sink', or 'bathtub')
        Returns:
            dict containing dimension constraints and mandatory requirements
        """
        if element_type not in self.element_dimensions:
            raise ValueError(f"Unknown element type: {element_type}")
        return self.element_dimensions[element_type]

    def validate_element_dimensions(self, element_type, width, depth, height):
        """
        Validate if dimensions are within allowed ranges
        """
        if element_type not in self.element_dimensions:
            return False, f"Unknown element type: {element_type}"

        dims = self.element_dimensions[element_type]
        
        if not (dims['width']['min'] <= width <= dims['width']['max']):
            return False, f"{element_type} width must be between {dims['width']['min']} and {dims['width']['max']} inches"
            
        if not (dims['depth']['min'] <= depth <= dims['depth']['max']):
            return False, f"{element_type} depth must be between {dims['depth']['min']} and {dims['depth']['max']} inches"
            
        if not (dims['height']['min'] <= height <= dims['height']['max']):
            return False, f"{element_type} height must be between {dims['height']['min']} and {dims['height']['max']} inches"
            
        return True, "Dimensions valid"

    def get_clearance_requirements(self, element_type):
        """
        Get clearance requirements including mandatory clear zones
        """
        if element_type not in self.clearances:
            raise ValueError(f"Unknown element type: {element_type}")
        return self.clearances[element_type]

    def get_mandatory_requirements(self, element_type):
        """
        Get mandatory requirements for element placement
        """
        if element_type not in self.element_dimensions:
            raise ValueError(f"Unknown element type: {element_type}")
        return self.element_dimensions[element_type]['mandatory']

    def validate_room_requirements(self, room_data):
        """
        Validate room against mandatory requirements
        """
        validation = {
            'valid': True,
            'messages': []
        }

        # Check room dimensions
        if room_data['boundaries']['max_x'] - room_data['boundaries']['min_x'] < self.mandatory_rules['minimum_room_width']:
            validation['valid'] = False
            validation['messages'].append(f"Room width must be at least {self.mandatory_rules['minimum_room_width']} inches")

        if room_data['boundaries']['max_y'] - room_data['boundaries']['min_y'] < self.mandatory_rules['minimum_room_depth']:
            validation['valid'] = False
            validation['messages'].append(f"Room depth must be at least {self.mandatory_rules['minimum_room_depth']} inches")

        return validation
