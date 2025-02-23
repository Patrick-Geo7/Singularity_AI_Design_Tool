from ortools.sat.python import cp_model
from constraints import BathroomConstraints
from visualization import BathroomVisualizer
from feasability_check import check_room_feasibility


class BathroomSolver:
    def __init__(self, room_width, room_length, door_x, door_y, door_width, 
                 has_toilet, has_sink, has_bathtub):
        self.room_width = room_width
        self.room_length = room_length
        self.door_x = door_x
        self.door_y = door_y
        self.door_width = door_width
        self.has_toilet = has_toilet
        self.has_sink = has_sink
        self.has_bathtub = has_bathtub
        self.constraints = BathroomConstraints()
        self.visualizer = BathroomVisualizer(room_width, room_length, door_x, door_y, door_width, fixture_images={'toilet': 'H:/Shared drives/AI Design Tool/00-PG_folder/03-Furniture AI Model/2nd Approach/Bathroom/Solver/Assets/2d_Images/toilet.png', 'sink': 'H:/Shared drives/AI Design Tool/00-PG_folder/03-Furniture AI Model/2nd Approach/Bathroom/Solver/Assets/2d_Images/sink.png', 'bathtub': 'H:/Shared drives/AI Design Tool/00-PG_folder/03-Furniture AI Model/2nd Approach/Bathroom/Solver/Assets/2d_Images/bathtub.png '})

    def _add_wall_placement_constraints(self, model, pos, walls, fixture_name):
        """Add wall placement constraints for a fixture with rotation consideration."""
        # Get base dimensions
        base_width, base_depth = self.constraints.FIXTURE_DIMENSIONS[fixture_name]
        side_clearance = self.constraints.CLEARANCES[fixture_name]['side']

        # Bottom wall placement (0° rotation)
        width_0, depth_0 = self.constraints.get_rotated_dimensions(fixture_name, 0)
        if depth_0 <= self.room_length:  # Ensure fixture fits
            model.Add(pos['y'] == 0).OnlyEnforceIf(walls['bottom'])
            model.Add(pos['x'] >= side_clearance).OnlyEnforceIf(walls['bottom'])
            model.Add(pos['x'] + width_0 <= self.room_width - side_clearance).OnlyEnforceIf(walls['bottom'])
        else:
            model.Add(walls['bottom'] == 0)  # Disable placement

        # Left wall placement (90° rotation)
        width_90, depth_90 = self.constraints.get_rotated_dimensions(fixture_name, 90)
        if depth_90 <= self.room_width:
            model.Add(pos['x'] == 0).OnlyEnforceIf(walls['left'])
            model.Add(pos['y'] >= side_clearance).OnlyEnforceIf(walls['left'])
            model.Add(pos['y'] + width_90 <= self.room_length - side_clearance).OnlyEnforceIf(walls['left'])
        else:
            model.Add(walls['left'] == 0)  # Disable placement

        # Top wall placement (180° rotation)
        width_180, depth_180 = self.constraints.get_rotated_dimensions(fixture_name, 180)
        if depth_180 <= self.room_length:
            model.Add(pos['y'] + depth_180 == self.room_length).OnlyEnforceIf(walls['top'])
            model.Add(pos['x'] >= side_clearance).OnlyEnforceIf(walls['top'])
            model.Add(pos['x'] + width_180 <= self.room_width - side_clearance).OnlyEnforceIf(walls['top'])
        else:
            model.Add(walls['top'] == 0)

        # Right wall placement (270° rotation)
        width_270, depth_270 = self.constraints.get_rotated_dimensions(fixture_name, 270)
        if depth_270 <= self.room_width:
            model.Add(pos['x'] + depth_270 == self.room_width).OnlyEnforceIf(walls['right'])
            model.Add(pos['y'] >= side_clearance).OnlyEnforceIf(walls['right'])
            model.Add(pos['y'] + width_270 <= self.room_length - side_clearance).OnlyEnforceIf(walls['right'])
        else:
            model.Add(walls['right'] == 0)  # Disable placement

    def _add_door_clearance_constraints(self, model, fixture, pos):
        """Enforce door clearance constraints for fixtures based on door position."""
        if not all(x is not None for x in [self.door_x, self.door_y, self.door_width]):
            return  # No door information provided

        door_clearance = self.door_width  # Clearance distance should match door width
        fixture_width, fixture_depth = self.constraints.FIXTURE_DIMENSIONS[fixture]

        is_in_door_range = self._create_door_range_constraint(model, pos,fixture)

        if self.door_y == 0:  # Bottom wall door
            model.Add(pos['y'] + fixture_depth >= door_clearance).OnlyEnforceIf(is_in_door_range)

        elif self.door_y == self.room_length:  # Top wall door
            model.Add(pos['y'] <= self.room_length - door_clearance - fixture_depth).OnlyEnforceIf(is_in_door_range)

        elif self.door_x == 0:  # Left wall door
            model.Add(pos['x'] + fixture_width >= door_clearance).OnlyEnforceIf(is_in_door_range)

        elif self.door_x == self.room_width:  # Right wall door
            model.Add(pos['x'] <= self.room_width - door_clearance - fixture_width).OnlyEnforceIf(is_in_door_range)

    def _create_door_range_constraint(self, model, pos, fixture):
        """Create a Boolean variable that checks if a fixture is in the door range."""
        is_start_side = model.NewBoolVar(f"{fixture}_is_start_side")
        is_end_side = model.NewBoolVar(f"{fixture}_is_end_side")

        fixture_width, fixture_depth = self.constraints.FIXTURE_DIMENSIONS[fixture]

        if self.door_x in [0, self.room_width]:  # Door on left/right wall
            model.Add(pos['y'] + fixture_depth >= self.door_y + self.door_width).OnlyEnforceIf(is_start_side)
            model.Add(pos['y'] <= self.door_y).OnlyEnforceIf(is_end_side)

        elif self.door_y in [0, self.room_length]:  # Door on top/bottom wall
            model.Add(pos['x'] + fixture_width >= self.door_x + self.door_width).OnlyEnforceIf(is_start_side)
            model.Add(pos['x'] <= self.door_x).OnlyEnforceIf(is_end_side)

        is_in_door_range = model.NewBoolVar(f"{fixture}_in_door_range")
        model.AddBoolOr([is_start_side, is_end_side]).OnlyEnforceIf(is_in_door_range)  # FIX: OR instead of AND

        return is_in_door_range

    def solve(self):
        """Create and solve the CP-SAT model for bathroom layout."""
        model = cp_model.CpModel()
        if not check_room_feasibility(self):
            print(f"🚨 Room {self.room_width}x{self.room_length} is too small to fit all fixtures!")
            return None

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

            pos = positions[fixture]
            walls = wall_bools[fixture]

            # Add door clearance constraints
            self._add_door_clearance_constraints(model,fixture,pos)
            # Add wall placement constraints considering rotation
            self._add_wall_placement_constraints(model, pos, walls, fixture)



            # Front clearances based on dataset rules
            # front_min = self.constraints.CLEARANCES[fixture]['front']

            # Add front clearance constraints for each wall
            # for wall, rotation in [('bottom', 0), ('left', 90), ('top', 180), ('right', 270)]:
            #     width, depth = self.constraints.get_rotated_dimensions(fixture, rotation)
            #
            #     if wall in ['bottom', 'top']:
            #         clear_var = model.NewIntVar(0, self.room_length, f'{fixture}_front_clear_{wall}')
            #         if wall == 'bottom':
            #             model.Add(clear_var <= self.room_length - (pos['y'] + depth)).OnlyEnforceIf(walls[wall])
            #         else:  # top
            #             model.Add(clear_var <= pos['y']).OnlyEnforceIf(walls[wall])
            #     else:  # left or right
            #         clear_var = model.NewIntVar(0, self.room_width, f'{fixture}_front_clear_{wall}')
            #         if wall == 'left':
            #             model.Add(clear_var <= self.room_width - (pos['x'] + depth)).OnlyEnforceIf(walls[wall])
            #         else:  # right
            #             model.Add(clear_var <= pos['x']).OnlyEnforceIf(walls[wall])
            #
            #     model.Add(clear_var >= front_min).OnlyEnforceIf(walls[wall])

        # Add fixture overlap prevention with proper rotation handling
        for i in range(len(fixtures)):
            for j in range(i + 1, len(fixtures)):
                f1, f2 = fixtures[i], fixtures[j]
                pos1, pos2 = positions[f1], positions[f2]
                walls1, walls2 = wall_bools[f1], wall_bools[f2]

                # Create separation variables
                sep_x = model.NewBoolVar(f'sep_x_{f1}_{f2}')
                sep_y = model.NewBoolVar(f'sep_y_{f1}_{f2}')

                # For each possible wall combination
                for wall1, rot1 in [('bottom', 0), ('left', 90), ('top', 180), ('right', 270)]:
                    for wall2, rot2 in [('bottom', 0), ('left', 90), ('top', 180), ('right', 270)]:
                        w1, d1 = self.constraints.get_rotated_dimensions(f1, rot1)
                        w2, d2 = self.constraints.get_rotated_dimensions(f2, rot2)

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
        solver.parameters.max_time_in_seconds = 120.0
        status = solver.Solve(model)

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            solution = {}
            for fixture in fixtures:
                pos = positions[fixture]
                # walls = wall_bools[fixture]

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