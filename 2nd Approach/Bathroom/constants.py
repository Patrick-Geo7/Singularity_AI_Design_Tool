bathroom_dataset = {
    "elements_rules": {
        "toilet": {
            "dimensions": {
                "width_range": [13.8, 17.7],  # inches
                "depth_range": [23.6, 29.5],
                "height_range": [13.8, 15.7]
            },
            "placement": {
                "wall_alignment": True,
                "min_wall_distance": 5.9,  # inches
                "clearance_front": 23.6,
                "clearance_sides": 7.9
            },
            "optional_attributes": ["water_connection", "ventilation"]
        },
        "sink": {
            "dimensions": {
                "width_range": [15.7, 47.2],
                "depth_range": [13.8, 23.6],
                "height_range": [33.5, 37.4]
            },
            "placement": {
                "wall_alignment": True,
                "min_wall_distance": 0,
                "clearance_front": 19.7,
                "clearance_sides": 5.9
            },
            "optional_attributes": ["water_connection", "mirror_above"]
        },
        "bathtub": {
            "dimensions": {
                "width_range": [27.6, 70.9],
                "depth_range": [55.1, 78.7],
                "height_range": [19.7, 23.6]
            },
            "placement": {
                "wall_alignment": True,
                "min_wall_distance": 0,
                "clearance_front": 23.6,
                "clearance_sides": 7.9,
                "corner_placement": "optional"
            },
            "optional_attributes": ["water_connection", "drain_location"]
        }
    },
    "spatial_relations": {
        "element_to_element": {
            "toilet_sink": {
                "min_distance": 11.8,
                "preferred_distance": 23.6,
                "max_distance": 59.1
            },
            "bathtub_toilet": {
                "min_distance": 11.8,
                "preferred_distance": 23.6
            },
            "bathtub_sink": {
                "min_distance": 11.8,
                "preferred_distance": 23.6
            }
        }
    }
}

# Add this to your existing constants.py
base_path = "G:\\Shared drives\\SINGULARITY\\02 - SNG Internal Documents\\06 - SNG Content\\00 - Content\\02 - SNG Content\\06 - Python Scripts\\Under Progress\\standaloneapp\\Trained_Model6\\assets"

element_assets = {
    "2d_symbols": {
        "toilet": f"{base_path}\\images\\toilet.png",
        "sink":  f"{base_path}\\images\\sink.png",
        "bathtub":  f"{base_path}\\images\\bathtub.png"
    },
    "3d_models": {
        "toilet": f"{base_path}\\images\\toilet.dae",
        "sink": f"{base_path}\\images\\sink.dae",
        "bathtub": f"{base_path}\\images\\bathtub.dae"
    }
}
