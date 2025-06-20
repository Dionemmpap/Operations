# This file provides an example of how to use the trajectory planning functionality, demonstrating a simple planning scenario.

import numpy as np
from src.utils.obstacles import get_obstacles
from src.utils.visualization import visualize_map
from src.planners.milp_planner import MILPTrajectoryPlanner

def main():
    map_boundary = [[0, 0], [10, 0], [10, 10], [0, 10]]
    obstacles = get_obstacles(map_boundary, 4)
    end_point = [9.9, 9.9]
    start_point = [0.1, 0.1]

    visualize_map(map_boundary, obstacles, {}, end_point)

    planner = MILPTrajectoryPlanner(map_boundary, obstacles, end_point, start_point, tau=0.2)
    trajectory = planner.plan_trajectory(horizon=5, max_iterations=20)

    if trajectory is not None:
        print("Planned trajectory:")
        for state in trajectory:
            print(state)

if __name__ == "__main__":
    main()