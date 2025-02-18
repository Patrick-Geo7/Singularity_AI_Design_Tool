from bathroom_solver import BathroomSolver
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_single_fixture():
    """Test with just a toilet in a simple room layout."""
    solver = BathroomSolver(
        room_width=60,  # Reasonably sized room
        room_length=72,
        door_x=0,      # Door on left wall
        door_y=30,
        door_width=30,
        has_toilet=True,
        has_sink=False,
        has_bathtub=False
    )

    try:
        solution = solver.solve()
        logger.info("Solution found for single fixture:")
        logger.info(solution)
        return True
    except Exception as e:
        logger.error(f"Single fixture test failed: {str(e)}")
        return False

def test_two_fixtures():
    """Test with toilet and sink."""
    solver = BathroomSolver(
        room_width=72,
        room_length=84,
        door_x=0,
        door_y=30,
        door_width=30,
        has_toilet=True,
        has_sink=True,
        has_bathtub=False
    )

    try:
        solution = solver.solve()
        logger.info("Solution found for two fixtures:")
        logger.info(solution)
        return True
    except Exception as e:
        logger.error(f"Two fixture test failed: {str(e)}")
        return False

def test_full_bathroom():
    """Test with all fixtures."""
    solver = BathroomSolver(
        room_width=96,   # 8 feet wide
        room_length=120, # 10 feet long
        door_x=0,
        door_y=30,
        door_width=30,
        has_toilet=True,
        has_sink=True,
        has_bathtub=True
    )

    try:
        solution = solver.solve()
        logger.info("Solution found for full bathroom:")
        logger.info(solution)
        return True
    except Exception as e:
        logger.error(f"Full bathroom test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("\nTesting single fixture layout...")
    single_success = test_single_fixture()

    if single_success:
        print("\nTesting two fixture layout...")
        two_success = test_two_fixtures()

        if two_success:
            print("\nTesting full bathroom layout...")
            full_success = test_full_bathroom()

    print("\nTest Results:")
    print(f"Single Fixture Test: {'PASSED' if single_success else 'FAILED'}")
    if single_success:
        print(f"Two Fixture Test: {'PASSED' if two_success else 'FAILED'}")
        if two_success:
            print(f"Full Bathroom Test: {'PASSED' if full_success else 'FAILED'}")