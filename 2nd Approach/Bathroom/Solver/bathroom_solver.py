from ortools.sat.python import cp_model
from constraints import BathroomConstraints
from visualization import BathroomVisualizer

class BathroomSolver:
    def __init__(self, room_width, room_length, door_x=None, door_y=None, door_width=None, 
                 has_toilet=True, has_sink=True, has_bathtub=True):
        self.room_width = room_width
        self.room_length = room_length
        self.door_x = door_x
        self.door_y = door_y
        self.door_width = door_width
        self.has_toilet = has_toilet
        self.has_sink = has_sink
        self.has_bathtub = has_bathtub
        self.constraints = BathroomConstraints()
        self.visualizer = BathroomVisualizer(room_width, room_length)

    def _add_wall_placement_constraints(self, model, pos, walls, width, depth, side_clearance):
        """Add wall placement constraints for a fixture."""
        # Bottom wall - fixture placed at y=0 facing up (0° rotation)
        model.Add(pos['y'] == 0).OnlyEnforceIf(walls['bottom'])
        model.Add(pos['x'] >= side_clearance).OnlyEnforceIf(walls['bottom'])
        model.Add(pos['x'] + width <= self.room_width - side_clearance).OnlyEnforceIf(walls['bottom'])

        # Left wall - fixture placed at x=0 facing right (90° rotation)
        model.Add(pos['x'] == 0).OnlyEnforceIf(walls['left'])
        model.Add(pos['y'] >= side_clearance).OnlyEnforceIf(walls['left'])
        model.Add(pos['y'] + width <= self.room_length - side_clearance).OnlyEnforceIf(walls['left'])

        # Top wall - fixture placed at y=room_length-depth facing down (180° rotation)
        model.Add(pos['y'] == self.room_length - depth).OnlyEnforceIf(walls['top'])
        model.Add(pos['x'] >= side_clearance).OnlyEnforceIf(walls['top'])
        model.Add(pos['x'] + width <= self.room_width - side_clearance).OnlyEnforceIf(walls['top'])

        # Right wall - fixture placed at x=room_width-depth facing left (270° rotation)
        model.Add(pos['x'] == self.room_width - depth).OnlyEnforceIf(walls['right'])
        model.Add(pos['y'] >= side_clearance).OnlyEnforceIf(walls['right'])
        model.Add(pos['y'] + width <= self.room_length - side_clearance).OnlyEnforceIf(walls['right'])

        # Add door clearance if provided
        if all(x is not None for x in [self.door_x, self.door_y, self.door_width]):
            door_clearance = 21  # inches from dataset rules

            # Door clearance based on door position
            if self.door_y == 0:  # Bottom wall door
                model.Add(pos['y'] >= door_clearance).OnlyEnforceIf(walls['bottom'])
            elif self.door_x == 0:  # Left wall door
                model.Add(pos['x'] >= door_clearance).OnlyEnforceIf(walls['left'])
            elif self.door_x == self.room_width:  # Right wall door
                model.Add(pos['x'] + width <= self.room_width - door_clearance).OnlyEnforceIf(walls['right'])
            elif self.door_y == self.room_length:  # Top wall door
                model.Add(pos['y'] + depth <= self.room_length - door_clearance).OnlyEnforceIf(walls['top'])

    def solve(self):
        """Create and solve the CP-SAT model for bathroom layout."""
        model = cp_model.CpModel()

        # Create variables for fixtures
        fixtures = []
        if self.has_toilet:
            fixtures.append('toilet')
        if self.has_sink:
            fixtures.append('sink')
        if self.has_bathtub:
            fixtures.append('bathtub')

        # Create variables for each fixture
        positions = {}
        wall_bools = {}

        # Handle fixture placement and constraints
        for fixture in fixtures:
            # Position variables
            positions[fixture] = {
                'x': model.NewIntVar(0, self.room_width, f'{fixture}_x'),
                'y': model.NewIntVar(0, self.room_length, f'{fixture}_y')
            }

            # Create boolean variables for each wall condition
            wall_bools[fixture] = {
                'bottom': model.NewBoolVar(f'{fixture}_bottom'),
                'left': model.NewBoolVar(f'{fixture}_left'),
                'top': model.NewBoolVar(f'{fixture}_top'),
                'right': model.NewBoolVar(f'{fixture}_right')
            }

            # Exactly one wall must be true
            model.AddExactlyOne(wall_bools[fixture].values())

            width, depth = self.constraints.FIXTURE_DIMENSIONS[fixture]
            pos = positions[fixture]
            walls = wall_bools[fixture]
            side_clearance = self.constraints.CLEARANCES[fixture]['side']

            # Add wall placement constraints with side clearances
            self._add_wall_placement_constraints(model, pos, walls, width, depth, side_clearance)

            # Front clearances based on dataset rules
            front_min = self.constraints.CLEARANCES[fixture]['front']

            # Bottom wall
            front_clear_bottom = model.NewIntVar(0, self.room_length, f'{fixture}_front_clear_bottom')
            model.Add(front_clear_bottom == self.room_length - (pos['y'] + depth)).OnlyEnforceIf(walls['bottom'])
            model.Add(front_clear_bottom >= front_min).OnlyEnforceIf(walls['bottom'])

            # Left wall
            front_clear_left = model.NewIntVar(0, self.room_width, f'{fixture}_front_clear_left')
            model.Add(front_clear_left == self.room_width - (pos['x'] + depth)).OnlyEnforceIf(walls['left'])
            model.Add(front_clear_left >= front_min).OnlyEnforceIf(walls['left'])

            # Top wall
            front_clear_top = model.NewIntVar(0, self.room_length, f'{fixture}_front_clear_top')
            model.Add(front_clear_top == pos['y']).OnlyEnforceIf(walls['top'])
            model.Add(front_clear_top >= front_min).OnlyEnforceIf(walls['top'])

            # Right wall
            front_clear_right = model.NewIntVar(0, self.room_width, f'{fixture}_front_clear_right')
            model.Add(front_clear_right == pos['x']).OnlyEnforceIf(walls['right'])
            model.Add(front_clear_right >= front_min).OnlyEnforceIf(walls['right'])

        # Add fixture overlap prevention
        for i in range(len(fixtures)):
            for j in range(i + 1, len(fixtures)):
                f1, f2 = fixtures[i], fixtures[j]
                pos1, pos2 = positions[f1], positions[f2]
                walls1, walls2 = wall_bools[f1], wall_bools[f2]

                # Create separation variables
                sep_x = model.NewBoolVar(f'sep_x_{f1}_{f2}')
                sep_y = model.NewBoolVar(f'sep_y_{f1}_{f2}')

                # For each possible wall combination
                for wall1 in ['bottom', 'top', 'left', 'right']:
                    for wall2 in ['bottom', 'top', 'left', 'right']:
                        # Get dimensions based on wall placement (rotation)
                        w1, d1 = self.constraints.get_rotated_dimensions(f1, 
                            0 if wall1 == 'bottom' else 
                            90 if wall1 == 'left' else 
                            180 if wall1 == 'top' else 270)
                        w2, d2 = self.constraints.get_rotated_dimensions(f2,
                            0 if wall2 == 'bottom' else 
                            90 if wall2 == 'left' else 
                            180 if wall2 == 'top' else 270)

                        # No overlap in x direction
                        model.Add(pos1['x'] + w1 <= pos2['x']).OnlyEnforceIf([sep_x, walls1[wall1], walls2[wall2]])
                        model.Add(pos2['x'] + w2 <= pos1['x']).OnlyEnforceIf([sep_x.Not(), walls1[wall1], walls2[wall2]])

                        # No overlap in y direction
                        model.Add(pos1['y'] + d1 <= pos2['y']).OnlyEnforceIf([sep_y, walls1[wall1], walls2[wall2]])
                        model.Add(pos2['y'] + d2 <= pos1['y']).OnlyEnforceIf([sep_y.Not(), walls1[wall1], walls2[wall2]])

                # Must be separated on at least one axis
                model.AddBoolOr([sep_x, sep_y])

        # Create solver and solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 60.0
        status = solver.Solve(model)

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            solution = {}
            for fixture in fixtures:
                pos = positions[fixture]
                walls = wall_bools[fixture]

                # Get wall placement and corresponding rotation
                wall = None
                for wall_name, wall_bool in wall_bools[fixture].items():
                    if solver.BooleanValue(wall_bool):
                        wall = wall_name
                        break

                # Map wall placement to rotation
                rotation_map = {
                    'bottom': 0,
                    'left': 90,
                    'top': 180,
                    'right': 270
                }
                rotation = rotation_map[wall]

                # Get actual position based on wall placement
                x = solver.Value(pos['x'])
                y = solver.Value(pos['y'])

                solution[fixture] = {
                    'x': x,
                    'y': y,
                    'rotation': rotation
                }

                print(f"\nDebug: {fixture}")
                print(f"  Position: ({x}, {y}) inches")
                print(f"  Wall: {wall}")
                print(f"  Rotation: {rotation}°")

            return solution
        else:
            raise ValueError("No valid solution found with given constraints")

    @classmethod
    def from_csv(cls, filepath):
        """Create solver instances from CSV data."""
        import pandas as pd
        df = pd.read_csv(filepath)
        instances = []
        for _, row in df.iterrows():
            solver = cls(
                room_width=row['Room_Width'],
                room_length=row['Room_Length'],
                door_x=row['Door_X_Position'],
                door_y=row['Door_Y_Position'],
                door_width=row['Door_Width'],
                has_toilet=bool(row['Has_Toilet']),
                has_sink=bool(row['Has_Sink']),
                has_bathtub=bool(row['Has_Bathtub'])
            )
            instances.append(solver)
        return instances