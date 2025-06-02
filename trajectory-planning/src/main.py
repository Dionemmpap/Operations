""" Receding Horizon Control for Trajectory Design """
import numpy as np
import heapq
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
from shapely.geometry import Point, Polygon, MultiPolygon, LineString
from planners.milp_planner import MILPTrajectoryPlanner
from planners.base_planner import TrajectoryDesignBase
from utils.visualization import visualize_map
from utils.obstacles import get_obstacles
from utils.map_loader import MapLoader
from pathlib import Path  
import json  

#---------------------------------------------------------------
# Main Function
#---------------------------------------------------------------

# def main():
#     """Main function to run the trajectory planning demo."""
#     # Setup environment
#     map_boundary = [[0, 0], [10, 0], [10, 10], [0, 10]]
#     obstacles = get_obstacles(map_boundary, 4)
#     end_point = [9.9, 9.9]
#     start_point = [0.1, 0.1]
    
#     # Visualize the environment
#     visualize_map(map_boundary, obstacles, {}, end_point)
    
#     # Initialize the MILP trajectory planner
#     planner = MILPTrajectoryPlanner(map_boundary, obstacles, end_point, start_point, tau=0.2)
#     planner.plan_trajectory(horizon=5, max_iterations=20)
    
#     # Visualize results
#     planner.plot(plt_traj=True)
#     planner.visualize_dynamics()



def main():
    """Main function to run the trajectory planning demo."""
    # Use MapLoader to load map data
    try:
        # Create map loader
        loader = MapLoader(maps_dir=Path(__file__).parent.parent / 'maps')
        
        # Load the scenarios.json file directly (since it's not in a subdirectory)
        map_path = Path(__file__).parent.parent / 'maps' / 'scenarios'/'paper_validation.json'
        
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
            use_hardcoded = False
            
            print(f"Loaded map: {map_data['name']}")
            print(f"Description: {map_data['description']}")
            
    except Exception as e:
        print(f"Error loading map: {e}")
        print("Falling back to hardcoded map values...")

    
    visualize_map(map_boundary=map_boundary, obstacles=obstacles, graph={}, end_point=end_point)
    # Initialize the base trajectory planner
    planner = TrajectoryDesignBase(map_boundary, obstacles, end_point, start_point, tau=0.1)
    planner.receding_horizon()
    planner.plot(plt_traj=True)


if __name__ == "__main__":
    main()