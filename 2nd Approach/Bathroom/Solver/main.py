import argparse
import pandas as pd
from bathroom_solver import BathroomSolver
from constraints import BathroomConstraints

def test_simple_case():
    """Run solver with a simple test case to verify core functionality."""
    print("\nRunning simple test case...")
    # Simple square room with door on bottom wall
    solver = BathroomSolver(
        room_width=100,
        room_length=100,
        door_x=35,
        door_y=0,
        door_width=30,
        has_toilet=True,
        has_sink=True,
        has_bathtub=False
    )

    try:
        print("\nAttempting to solve layout...")
        solution = solver.solve()
        print("\nTest case solution found!")
        print("\nFixture positions and rotations:")
        for fixture, pos in solution.items():
            print(f"\n{fixture.capitalize()}:")
            print(f"  Position: ({pos['x']}, {pos['y']}) inches")
            print(f"  Rotation: {pos['rotation']}°")
            if pos['rotation'] in [0, 180]:  # Vertical orientation
                print("  Aligned: Vertical (parallel to width)")
            else:  # 90 or 270 degrees
                print("  Aligned: Horizontal (parallel to length)")

        print("\nGenerating visualization...")
        solver.visualizer.render(solution)
        print("Visualization has been saved to 'bathroom_layout_test.png'")
        return True
    except Exception as e:
        print(f"\nTest case failed: {str(e)}")
        print("Detailed error information:")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description='Bathroom Fixture Placement Solver')
    parser.add_argument('--csv', type=str, default='H:/Shared drives/AI Design Tool/00-PG_folder/03-Furniture AI Model/Data/preprocessed/X_train.csv',
                      help='Path to input CSV file (default: X_train.csv)')
    parser.add_argument('--index', type=int,
                      help='Specific room index to solve (optional)')
    parser.add_argument('--test', action='store_true',
                      help='Run simple test case instead of CSV data')
    args = parser.parse_args()

    if args.test:
        success = test_simple_case()
        if not success:
            print("\nSimple test case failed. Please fix core functionality before processing real data.")
            return

    try:
        # Load data from CSV
        print(f"\nLoading data from {args.csv}")
        df = pd.read_csv(args.csv)

        if args.index is not None:
            if args.index >= len(df):
                raise ValueError(f"Invalid index {args.index}. File contains {len(df)} rooms.")
            df = df.iloc[args.index:args.index+1]

        solved_count = 0
        total_rooms = len(df)
        print(f"\nProcessing {total_rooms} room{'s' if total_rooms > 1 else ''}...")

        for index, row in df.iterrows():
            print(f"\nSolving room {index}")
            print(f"Room dimensions: {row['Room_Length']}x{row['Room_Width']} inches")
            print(f"Door position: ({row['Door_X_Position']}, {row['Door_Y_Position']}) inches")
            print(f"Door width: {row['Door_Width']} inches")
            print("Fixtures required: " + 
                  f"Toilet({'Yes' if row['Has_Toilet'] else 'No'}), " +
                  f"Sink({'Yes' if row['Has_Sink'] else 'No'}), " +
                  f"Bathtub({'Yes' if row['Has_Bathtub'] else 'No'})")

            try:
                # Create and run solver for this room
                solver = BathroomSolver(
                    room_width=row['Room_Width'],
                    room_length=row['Room_Length'],
                    door_x=row['Door_X_Position'],
                    door_y=row['Door_Y_Position'],
                    door_width=row['Door_Width'],
                    has_toilet=bool(row['Has_Toilet']),
                    has_sink=bool(row['Has_Sink']),
                    has_bathtub=bool(row['Has_Bathtub'])
                )

                print("\nApplying solver with constraints:")
                print("- Fixtures must be aligned to walls")
                print("- Toilet requires 9\" side clearance")
                print("- All fixtures require proper front clearance")
                print("- No fixture overlap allowed")
                print("- Door swing clearance maintained")

                solution = solver.solve()
                print("\nSolution found!")
                print("\nFixture positions and rotations:")
                for fixture, pos in solution.items():
                    print(f"\n{fixture.capitalize()}:")
                    print(f"  Position: ({pos['x']}, {pos['y']}) inches")
                    print(f"  Rotation: {pos['rotation']}°")
                    if pos['rotation'] in [0, 180]:  # Vertical orientation
                        print("  Aligned: Vertical (parallel to width)")
                    else:  # 90 or 270 degrees
                        print("  Aligned: Horizontal (parallel to length)")


                print("\nGenerating visualization...")
                solver.visualizer.render(solution,index)
                if index == 3:
                    solver.visualizer.render_3D_plotly(solution)
                print(f"Visualization has been saved to 'bathroom_layout_{index}.png'")
                solved_count += 1

            except ValueError as e:
                print(f"\nCould not solve room {index}: {str(e)}")
                if "No valid solution found" in str(e):
                    print("\nPossible reasons:")
                    print("- Room dimensions too small for required fixtures")
                    print("- Clearance requirements cannot be met")
                    print("- Door placement restricts valid fixture positions")
            except Exception as e:
                print(f"\nError processing room {index}: {str(e)}")
                print("Detailed error information:")
                import traceback
                traceback.print_exc()

        print(f"\nProcessing complete. Successfully solved {solved_count} out of {total_rooms} rooms.")
        if solved_count < total_rooms:
            print("\nSome rooms could not be solved. Review the output above for specific error messages.")
        else:
            print("\nAll rooms successfully solved!")

    except FileNotFoundError:
        print(f"\nError: Could not find input file '{args.csv}'")
    except pd.errors.EmptyDataError:
        print(f"\nError: Input file '{args.csv}' is empty")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {str(e)}")
        print("Detailed error information:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()