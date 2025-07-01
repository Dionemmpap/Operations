# trajectory-planning/src/main.py
import matplotlib.pyplot as plt
from utils.visualization import visualize_map
from utils.obstacles import get_obstacles
from utils.map_loader import MapLoader
from pathlib import Path  
import json  
from planners.planner_chapter2 import RecedingHorizonController


def main():
    """Main function to run the trajectory planning demo."""
    loader = MapLoader(maps_dir=Path(__file__).parent.parent / 'maps'/'scenarios')
    # Use MapLoader to load map data
    available_maps = loader.list_available_maps()
    # Display available maps
    print("\n=== Available Maps ===")
    for i, map_info in enumerate(available_maps, 1):
        print(f"{i}. {map_info['name']} - {map_info['description']}")
    
    try:
        selection = input("\nSelect a map by number (default is 1): ")
        if not selection.strip():
            selection = '1'
        selected_index = int(selection) - 1
        if selected_index < 0 or selected_index >= len(available_maps):
            raise ValueError("Invalid selection.")
    except ValueError as e:
        print(f"Error: {e}. Defaulting to map 1.")
        selected_index = 0
    map_data = loader.load_map(available_maps[selected_index]['name'])
    
    # Extract map data
    map_boundary = map_data['map_boundary']
    obstacles = map_data['obstacles']
    end_point = map_data['end_point']
    start_point = map_data['start_point']

    print(f"Loaded map: {map_data['name']}")
    print(f"Description: {map_data['description']}")

    visualize_map(map_boundary=map_boundary, obstacles=obstacles, graph={}, end_point=end_point)

    controller = RecedingHorizonController(map_boundary, obstacles, start_point, end_point,tau=0.5,N=15, Ne=6, umax=1)
    controller.plan_and_execute()
    controller.plot_results()


if __name__ == "__main__":
    main()