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
    # Define map boundary and obstacles
    #map_boundary = [[0, 0], [10, 0], [10, 10], [0, 10]]
    #obstacles = get_obstacles(map_boundary, 4)
    #If you'd like a custom obstacle, you can add it here
    #obstacles.append([[5, 5], [6, 5], [6, 6], [5, 6]])
    #end_point = [9.9, 9.9]

    #visualize_map(map_boundary, obstacles, {}, end_point)

    #td = TrajectoryDesign(map_boundary, obstacles, end_point, [0, 0], 0.1)
#
    #td.receding_horizon()
    #td.plot(plt_traj=True)
    #
    ## # Print distances for each obstacle corner to the endpoint
    #print("Distances from obstacle corners to endpoint:")
    #for obstacle in td.obstacles:
    #    for corner in obstacle:
    #        print(f"Corner {corner} -> Endpoint: {td.distances[tuple(corner)]:.2f}")
   
    #Validation of paper results
    map_boundary = [[0, 0], [0, 10], [45, 10], [45, 0]]
    obstacles = []
    # Obstacles
    obstacles.append([[1, 1], [2, 1], [2, 4.5], [1, 4.5]])

    obstacles.append([[1, 6], [5, 6], [5, 7], [1, 7]])
    obstacles.append([[4, 4], [5, 4], [5, 7], [4, 7]])
    obstacles.append([[5, 6.5], [7, 6.5], [7, 10], [5, 10]])

    obstacles.append([[6, 3], [7, 3], [7, 5], [6, 5]])
    obstacles.append([[6, 0.2], [7, 0.2], [7, 2], [6, 2]])
    obstacles.append([[6, 0.2], [10, 0.2], [10, 1], [6, 1]])
    obstacles.append([[6, 4], [10, 4], [10, 5], [6, 5]])
    obstacles.append([[9, 0.2], [10, 0.2], [10, 5], [9, 5]])

    obstacles.append([[11.8, 7], [14, 7], [14, 9], [11.8, 9]])
    obstacles.append([[11, 4], [13.5, 4], [13.5, 6], [11, 6]])
    obstacles.append([[11, 1], [13.5, 1], [13.5, 3], [11, 3]])

    obstacles.append([[15, 0], [18, 0], [18, 9], [15, 9]])
    
    obstacles.append([[22, 2], [25, 2], [25, 3], [22, 3]])

    obstacles.append([[24, 7], [27, 7], [27, 10], [24, 10]])

    obstacles.append([[29.5, 7], [31, 7], [31, 8.5], [29.5, 8.5]])

    obstacles.append([[28, 1], [33, 1], [33, 4], [28, 4]])

    obstacles.append([[35, 5], [38, 5], [38, 8], [35, 8]])
    obstacles.append([[37, 3], [40, 3], [40, 6], [37, 6]])
    obstacles.append([[39, 1], [42, 1], [42, 4], [39, 4]])

    end_point = [44.9, 5]

    visualize_map(map_boundary, obstacles, {}, end_point)

    td = TrajectoryDesignBase(map_boundary, obstacles, end_point, [0, 5], 0.1)

    td.receding_horizon()
    td.plot(plt_traj=True)
    
    # # Print distances for each obstacle corner to the endpoint
    print("Distances from obstacle corners to endpoint:")
    for obstacle in td.obstacles:
        for corner in obstacle:
            print(f"Corner {corner} -> Endpoint: {td.distances[tuple(corner)]:.2f}")


if __name__ == "__main__":
    main()