class BathroomConstraints:
    def __init__(self):
        # Exact dimensions from rules
        self.FIXTURE_DIMENSIONS = {
            'toilet': (19, 28),  # Width x Depth
            'sink': (30, 20),    # Width x Depth  
            'bathtub': (30, 60)  # Width x Depth
        }

        # Minimum clearances in inches
        self.CLEARANCES = {
            'toilet': {
                'front': 21,  # Minimum front clearance
                'side': 9     # Minimum side clearance (9" on both sides)
            },
            'sink': {
                'front': 21,  # Minimum front clearance
                'side': 4     # Minimum side clearance
            },
            'bathtub': {
                'front': 21,  # Minimum front clearance
                'side': 4     # Minimum side clearance
            }
        }

        # Rotation angles (in degrees) 
        self.VALID_ROTATIONS = [0, 90, 180, 270]  # 0=up, 90=right, 180=down, 270=left

    def get_fixture_position(self, fixture_type, rotation, room_width, room_length, x, y):
        """Calculate actual position based on rotation and wall placement rules"""
        width, depth = self.FIXTURE_DIMENSIONS[fixture_type]

        if rotation == 0:  # Bottom wall, facing up
            return (x, 0)
        elif rotation == 90:  # Left wall, facing right
            return (0, y)
        elif rotation == 180:  # Top wall, facing down
            return (x, room_length - depth)
        elif rotation == 270:  # Right wall, facing left 
            return (room_width - depth, y)
        return None

    def get_clearance_requirements(self, fixture_type, rotation, room_width, room_length, x, y):
        """Calculate clearance requirements based on wall placement"""
        width, depth = self.FIXTURE_DIMENSIONS[fixture_type]
        front_min = self.CLEARANCES[fixture_type]['front']
        side_min = self.CLEARANCES[fixture_type]['side']

        # Calculate front clearance based on wall placement rules from dataset
        if rotation == 0:  # Bottom wall
            front_clearance = room_length - (y + depth)
        elif rotation == 90:  # Left wall
            front_clearance = room_width - (x + depth)
        elif rotation == 180:  # Top wall
            front_clearance = room_length - depth  # Top wall rule from dataset
        else:  # Right wall (270)
            front_clearance = room_width - depth  # Right wall rule from dataset

        return {
            'front': max(front_clearance, front_min),
            'side': side_min
        }

    def get_rotated_dimensions(self, fixture_type, rotation):
        """Get dimensions after rotation, following the parallel/perpendicular rules"""
        width, depth = self.FIXTURE_DIMENSIONS[fixture_type]
        # For 90° and 270° rotations, width is parallel to room length
        return (depth, width) if rotation in [90, 270] else (width, depth)