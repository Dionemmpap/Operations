import matplotlib.pyplot as plt
from gurobipy import Model, GRB, quicksum

class TrajectoryDesign:
    """Class to design a trajectory using receding horizon control."""
    def __init__(self, map_boundary, obstacles, end_point, start_point):
        """
        Initializes the trajectory design class.
        
        :param map_boundary: List of corner points defining the map boundary (e.g., [[x1, y1], [x2, y2], ...]).
        :param obstacles: List of polygons defining obstacles (each polygon is a list of corner points).
        :param end_point: Coordinates of the goal position [x_goal, y_goal].
        :param start_point: Initial state of the system [x, y, x_dot, y_dot].
        """
        self.map_boundary = map_boundary
        self.obstacles = obstacles
        self.end_point = end_point
        self.start_point = start_point

    def solve_trajectory_optimization(self, m, v_max, u_max, N, M=1e6):
        """Solve the trajectory optimization problem using Gurobi."""
        model = Model("Trajectory Optimization")

        # Variables
        x = model.addVars(N + 1, 4, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="x")  # States [x, y, x_dot, y_dot]
        u = model.addVars(N, 2, lb=-u_max, ub=u_max, name="u")  # Controls [u_x, u_y]
        b_goal = model.addVars(N, vtype=GRB.BINARY, name="b_goal")  # Goal indicator binary variables
        b_obst = model.addVars(N, len(self.obstacles), vtype=GRB.BINARY, name="b_obst")  # Obstacle indicator binary variables
        s_u = model.addVars(2, lb=0, name="s_u")  # Slack variables for L_1 norm
        s_v = model.addVars(2, lb=0, name="s_v")  # Slack variables for L_1 norm

        # Initial state constraint
        for i in range(4):
            model.addConstr(x[0, i] == self.start_point[i], name=f"initial_state_{i}")

        # Goal constraints
        for i in range(N):
            model.addConstr(x[i + 1, 0] - self.end_point[0] <= M * (1 - b_goal[i]), name=f"goal_x_upper_{i}")
            model.addConstr(x[i + 1, 0] - self.end_point[0] >= -M * (1 - b_goal[i]), name=f"goal_x_lower_{i}")
            model.addConstr(x[i + 1, 1] - self.end_point[1] <= M * (1 - b_goal[i]), name=f"goal_y_upper_{i}")
            model.addConstr(x[i + 1, 1] - self.end_point[1] >= -M * (1 - b_goal[i]), name=f"goal_y_lower_{i}")
        model.addConstr(quicksum(b_goal[i] for i in range(N)) == 1, name="goal_sum")

        # Obstacle constraints (approximate by bounding box)
        for t in range(N):
            for j, obstacle in enumerate(self.obstacles):
                x_min = min(pt[0] for pt in obstacle)
                x_max = max(pt[0] for pt in obstacle)
                y_min = min(pt[1] for pt in obstacle)
                y_max = max(pt[1] for pt in obstacle)

                model.addConstr(x[t + 1, 0] <= x_min + M * b_obst[t, j], name=f"obst_x_low_{t}_{j}")
                model.addConstr(x[t + 1, 0] >= x_max - M * b_obst[t, j], name=f"obst_x_high_{t}_{j}")
                model.addConstr(x[t + 1, 1] <= y_min + M * b_obst[t, j], name=f"obst_y_low_{t}_{j}")
                model.addConstr(x[t + 1, 1] >= y_max - M * b_obst[t, j], name=f"obst_y_high_{t}_{j}")
            model.addConstr(quicksum(b_obst[t, j] for j in range(len(self.obstacles))) <= 3, name=f"obst_sum_{t}")

        # Dynamics constraints
        for t in range(N):
            model.addConstrs((
                x[t + 1, i] == x[t, i] + x[t, i + 2] for i in range(2)), name=f"dynamics_pos_{t}")
            model.addConstrs((
                x[t + 1, i + 2] == x[t, i + 2] + u[t, i] / m for i in range(2)), name=f"dynamics_vel_{t}")
            model.addConstr(quicksum(x[t, i + 2] ** 2 for i in range(2)) <= v_max ** 2, name=f"speed_limit_{t}")
            model.addConstr(quicksum(u[t, i] ** 2 for i in range(2)) <= u_max ** 2, name=f"control_limit_{t}")

        # Objective function
        model.addConstr(s_u[0] >= x[N, 0] - self.end_point[0], name="slack_su1")
        model.addConstr(s_u[0] >= -(x[N, 0] - self.end_point[0]), name="slack_su2")
        model.addConstr(s_v[0] >= x[N, 1] - self.end_point[1], name="slack_sv1")
        model.addConstr(s_v[0] >= -(x[N, 1] - self.end_point[1]), name="slack_sv2")
        model.setObjective(s_u.sum() + s_v.sum(), GRB.MINIMIZE)

        # Solve the problem
        model.optimize()

        # Extract solution
        if model.status == GRB.OPTIMAL:
            trajectory = {
                "positions": [(x[t, 0].X, x[t, 1].X) for t in range(N + 1)],
                "velocities": [(x[t, 2].X, x[t, 3].X) for t in range(N + 1)],
                "controls": [(u[t, 0].X, u[t, 1].X) for t in range(N)]
            }
            return trajectory
        else:
            raise ValueError("Optimization did not converge.")

    def plot_trajectory(self, trajectory):
        """Plot the obstacle map and the trajectory."""
        fig, ax = plt.subplots(figsize=(8, 8))

        # Plot map boundaries
        map_x, map_y = zip(*self.map_boundary)
        ax.plot(map_x + (map_x[0],), map_y + (map_y[0],), "b-", label="Map Boundary")

        # Plot obstacles
        for obstacle in self.obstacles:
            obs_x, obs_y = zip(*obstacle)
            ax.fill(obs_x + (obs_x[0],), obs_y + (obs_y[0],), "red", alpha=0.5, label="Obstacle")

        # Plot start and end points
        ax.plot(self.start_point[0], self.start_point[1], "go", label="Start Point")
        ax.plot(self.end_point[0], self.end_point[1], "bo", label="End Point")

        # Plot trajectory
        positions = trajectory["positions"]
        x_vals, y_vals = zip(*positions)
        ax.plot(x_vals, y_vals, "k-", label="Trajectory")

        ax.legend()
        plt.grid(True)
        plt.show()


# Main Function
def main():
    # Define map boundary and obstacles
    map_boundary = [[0, 0], [10, 0], [10, 10], [0, 10]]
    obstacles = [
        [[2, 2], [4, 2], [4, 4], [2, 4]],  # Obstacle 1
        [[6, 6], [8, 6], [8, 8], [6, 8]],  # Obstacle 2
    ]
    end_point = [9, 9]
    start_point = [1, 1, 0, 0]  # Initial state: [x, y, x_dot, y_dot]

    # Create the trajectory design object
    td = TrajectoryDesign(map_boundary, obstacles, end_point, start_point)

    # Solve the trajectory optimization problem
    trajectory = td.solve_trajectory_optimization(m=150.0, v_max=250.0, u_max=1000.0, N=12)

    # Plot the obstacle map and trajectory
    td.plot_trajectory(trajectory)


if __name__ == "__main__":
    main()
