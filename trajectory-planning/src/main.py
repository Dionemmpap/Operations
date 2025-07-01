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

    # Use MapLoader to load map data
    loader = MapLoader(maps_dir=Path(__file__).parent.parent / 'maps'/'scenarios')
    map_data = loader.load_map('paper_validation')
    
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