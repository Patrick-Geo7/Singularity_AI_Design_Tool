from constants import element_dimensions, clearances, mandatory_rules
print("Loaded constants:", element_dimensions.keys())

from dimension_rules import DimensionRules
from layout_manager import PlacementValidator,BathroomLayout
from preview_handler import DimensionHandler, PreviewHandler

class BathroomLayoutCollector:
    def __init__(self):
        # Initialize in correct order with dependencies
        self.dimension_rules = DimensionRules()
        self.bathroom_layout = BathroomLayout(self.dimension_rules)
        self.placement_validator = PlacementValidator(
            dimension_rules=self.dimension_rules,
            bathroom_layout=self.bathroom_layout
        )
        self.dimension_handler = DimensionHandler()
        self.preview_handler = PreviewHandler()


    def generate_trial_layout(self, room_data):
        print("Starting layout generation...")  # Debug print
        result = {
            'success': False,
            'layout': None,
            'messages': [],
            'preview': None
        }

        try:
            # Define bathroom elements in priority order
            elements_to_place = ['toilet', 'sink', 'bathtub']
            placed_elements = []

            for element_type in elements_to_place:
                print(f"Attempting to place {element_type}")  # Debug print
                placement = self._place_element(
                    element_type,
                    room_data,
                    placed_elements
                )

                if placement['success']:
                    placed_elements.append(placement['element'])
                else:
                    result['messages'].append(f"Failed to place {element_type}: {placement['message']}")
                    return result

            layout = {
                'room_data': room_data,
                'elements': placed_elements
            }

            result.update({
                'success': True,
                'layout': layout
            })

        except Exception as e:
            result['messages'].append(f"Error generating layout: {str(e)}")

        return result

    def _place_element(self, element_type, room_data, existing_elements):
        print(f"Placing element: {element_type}")  # Debug print
        result = {
            'success': False,
            'element': None,
            'message': ''
        }

        try:
            # Get valid positions
            valid_positions = self.bathroom_layout.calculate_valid_positions(
                element_type,
                room_data,
                existing_elements
            )

            if not valid_positions:
                result['message'] = f"No valid positions found for {element_type}"
                return result

            # Use best position (highest score)
            best_position = valid_positions[0]
            
            # Get element dimensions
            element_dims = self.dimension_rules.get_element_dimensions(element_type)
            print(f"Element dimensions: {element_dims}")  # Debug print

            # Create element data
            element = {
                'type': element_type,
                'position': best_position['position'],
                'dimensions': {
                    'width': element_dims['width']['min'],
                    'depth': element_dims['depth']['min'],
                    'height': element_dims['height']['min']
                }
            }

            result['success'] = True
            result['element'] = element

        except Exception as e:
            result['message'] = str(e)

        return result

if __name__ == "__main__":
    # Test room data
    room_data = {
        'boundaries': {
            'min_x': 0,
            'max_x': 120,  # 8 feet
            'min_y': 0,
            'max_y': 144  # 10 feet
        },
        'walls': [
            {'position': 0, 'orientation': 'vertical', 'plumbing': True},
            {'position': 120, 'orientation': 'vertical', 'plumbing': False},
            {'position': 0, 'orientation': 'horizontal', 'plumbing': True},
            {'position': 144, 'orientation': 'horizontal', 'plumbing': False}
        ]
    }

    # Create collector and generate trial layout
    collector = BathroomLayoutCollector()
    trial_layout = collector.generate_trial_layout(room_data)

    # Print results
    if trial_layout['success']:
        print("Layout generated successfully!")
        print("Generated layout:", trial_layout['layout'])
    else:
        print("Layout generation failed:")
        for message in trial_layout['messages']:
            print(f"- {message}")

