# trajectory-planning/src/sensitivity.py
import matplotlib.pyplot as plt

from utils.visualization import visualize_map
from utils.obstacles import get_obstacles
from utils.map_loader import MapLoader
from pathlib import Path  
import json  
from planners.planner_chapter2 import RecedingHorizonController


def main():
    """Main function to run the trajectory planning demo."""
    # Use MapLoader to load map data
    try:
        # Create map loader
        loader = MapLoader(maps_dir=Path(__file__).parent.parent / 'maps')
        
        # Load the scenarios.json file directly (since it's not in a subdirectory)
        map_path = Path(__file__).parent.parent / 'maps' / 'scenarios'/ 'sensitivity analysis'/ 'baseline_map_sa.json'
        
        # Check if file exists
        if not map_path.exists():
            print(f"Warning: Map file not found at {map_path}")
        else:
            # Load map data from file
            with open(map_path, 'r') as f:
                import json
                map_data = json.load(f)
                
            # Extract map data
            map_boundary = map_data['map_boundary']
            obstacles = map_data['obstacles']
            end_point = map_data['end_point']
            start_point = map_data['start_point']
            
            print(f"Loaded map: {map_data['name']}")
            print(f"Description: {map_data['description']}")
            
    except Exception as e:
        print(f"Error loading map: {e}")
        print("Falling back to hardcoded map values...")

    
    visualize_map(map_boundary=map_boundary, obstacles=obstacles, graph={}, end_point=end_point)

    controller = RecedingHorizonController(map_boundary, obstacles, start_point, end_point)
    controller.plan_and_execute()
    controller.plot_results()


if __name__ == "__main__":
    main()