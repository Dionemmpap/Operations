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


# def main():
#     # Define map boundary and obstacles
#     #map_boundary = [[0, 0], [10, 0], [10, 10], [0, 10]]
#     #obstacles = get_obstacles(map_boundary, 4)
#     #If you'd like a custom obstacle, you can add it here
#     #obstacles.append([[5, 5], [6, 5], [6, 6], [5, 6]])
#     #end_point = [9.9, 9.9]

#     #visualize_map(map_boundary, obstacles, {}, end_point)

#     #td = TrajectoryDesign(map_boundary, obstacles, end_point, [0, 0], 0.1)
# #
#     #td.receding_horizon()
#     #td.plot(plt_traj=True)
#     #
#     ## # Print distances for each obstacle corner to the endpoint
#     #print("Distances from obstacle corners to endpoint:")
#     #for obstacle in td.obstacles:
#     #    for corner in obstacle:
#     #        print(f"Corner {corner} -> Endpoint: {td.distances[tuple(corner)]:.2f}")
   
#     #Validation of paper results
#     map_boundary = [[0, 0], [0, 10], [45, 10], [45, 0]]
#     obstacles = []
#     # Obstacles
#     obstacles.append([[1, 1], [2, 1], [2, 4.5], [1, 4.5]])

#     obstacles.append([[1, 6], [5, 6], [5, 7], [1, 7]])
#     obstacles.append([[4, 4], [5, 4], [5, 7], [4, 7]])
#     obstacles.append([[5, 6.5], [7, 6.5], [7, 10], [5, 10]])

#     obstacles.append([[6, 3], [7, 3], [7, 5], [6, 5]])
#     obstacles.append([[6, 0.2], [7, 0.2], [7, 2], [6, 2]])
#     obstacles.append([[6, 0.2], [10, 0.2], [10, 1], [6, 1]])
#     obstacles.append([[6, 4], [10, 4], [10, 5], [6, 5]])
#     obstacles.append([[9, 0.2], [10, 0.2], [10, 5], [9, 5]])

#     obstacles.append([[11.8, 7], [14, 7], [14, 9], [11.8, 9]])
#     obstacles.append([[11, 4], [13.5, 4], [13.5, 6], [11, 6]])
#     obstacles.append([[11, 1], [13.5, 1], [13.5, 3], [11, 3]])

#     obstacles.append([[15, 0], [18, 0], [18, 9], [15, 9]])
    
#     obstacles.append([[22, 2], [25, 2], [25, 3], [22, 3]])

#     obstacles.append([[24, 7], [27, 7], [27, 10], [24, 10]])

#     obstacles.append([[29.5, 7], [31, 7], [31, 8.5], [29.5, 8.5]])

#     obstacles.append([[28, 1], [33, 1], [33, 4], [28, 4]])

#     obstacles.append([[35, 5], [38, 5], [38, 8], [35, 8]])
#     obstacles.append([[37, 3], [40, 3], [40, 6], [37, 6]])
#     obstacles.append([[39, 1], [42, 1], [42, 4], [39, 4]])

#     end_point = [44.9, 5]

#     visualize_map(map_boundary, obstacles, {}, end_point)

#     td = TrajectoryDesignBase(map_boundary, obstacles, end_point, [0, 5], 0.1)

#     td.receding_horizon()
#     td.plot(plt_traj=True)
    
#     # # Print distances for each obstacle corner to the endpoint
#     print("Distances from obstacle corners to endpoint:")
#     for obstacle in td.obstacles:
#         for corner in obstacle:
#             print(f"Corner {corner} -> Endpoint: {td.distances[tuple(corner)]:.2f}")



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