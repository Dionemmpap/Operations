import numpy as np
from functools import partial

import parameters as p
from main import get_obstacles, StandardTrajectoryPlanner, MILPTrajectoryPlanner

def deviate_point(point, shift_values, times_per_value=1, seed=None):
    np.random.seed(seed)
    list_deviated_points = [point]
    for mean, std in shift_values:
        for i in range(times_per_value):
            deviation = np.random.normal(mean, std, size=2)
            list_deviated_points.append([point[0] + deviation[0], point[1] + deviation[1]])
    return list_deviated_points


def deviate_obstacles(obstacles, deviation_values, times_per_value=1, seed=None):
    np.random.seed(seed)
    list_deviated_obstacles = [obstacles]
    for mean, std in deviation_values:
        for i in range(times_per_value):
            deviation = np.random.normal(mean, std, size=(len(obstacles), 2))
            deviated_obstacles = [[obstacles[i][0] + deviation[i][0], obstacles[i][1] + deviation[i][1]] for i in len(obstacles)]
            list_deviated_obstacles.append(deviated_obstacles)
    return list_deviated_obstacles


def run_mains(number_of_obstacles, tau, planner_type=1, horizon=5, max_iteration=1, func_point_deviation=None, func_obstacle_deviation=None, plt_traj=False):
           
    for num in number_of_obstacles:
        print(f"\nNumber of Obstacles: {num}")
        map_boundary = [[0, 0], [10, 0], [10, 10], [0, 10]]
        obstacles = get_obstacles(map_boundary, num)
        end_point = [9.9, 9.9]
        start_point = [0.1, 0.1]

        if func_obstacle_deviation is not None:
            print("Deviating obstacles...")
            list_obstacles = func_obstacle_deviation(obstacles)
        else:
            list_obstacles = [obstacles]

        if func_point_deviation is not None:
            print("Deviating start and end points...")
            list_start_points = func_point_deviation(start_point)
            list_end_points = func_point_deviation(end_point)
        else:
            list_start_points = [start_point]
            list_end_points = [end_point]
            
        for obstacles in list_obstacles:
            print(f"Using obstacles: {obstacles}")

            for start_point, end_point in zip(list_start_points, list_end_points):
                print(f"Using Start Point: {start_point}, End Point: {end_point}")

                if planner_type == 1:
                    print("\nUsing Standard Trajectory Planner")
                    for t in tau:
                        print(f"Using tau: {t}")
                        # Initialize the planner
                        planner = StandardTrajectoryPlanner(map_boundary, obstacles, end_point, start_point, tau=t)
                        planner.plan_trajectory()

                        # Visualize results
                        planner.plot(plt_traj=plt_traj)
                        planner.visualize_dynamics()

                else:
                    print("\nUsing MILP Trajectory Planner with Dynamic Feasibility")
                    for t in tau:
                        for h in horizon:
                            for m_i in max_iteration:
                                print(f"Using tau: {t}, horizon: {h}, max_iterations: {m_i}")
                                # Initialize the planner
                                planner = MILPTrajectoryPlanner(map_boundary, obstacles, end_point, start_point, tau=t)
                                planner.plan_trajectory(horizon=h, max_iterations=m_i)

                                # Visualize results
                                planner.plot(plt_traj=plt_traj)
                                planner.visualize_dynamics()


if __name__ == "__main__":
    if p.obstacle_deviation:
        print("Running with obstacle deviation")
        deviate_obstacles_func = partial(deviate_obstacles, times_per_value=p.obstacle_deviations_times, deviation_values=p.obstacle_deviations_values)
    else:
        print("Running without obstacle deviation")
        deviate_obstacles_func = None
    run_mains(p.num_obstacles, p.tau, p.planner_type, p.horizon, p.iteration, func_obstacle_deviation=deviate_obstacles_func, plt_traj=p.plot_trajectory)
