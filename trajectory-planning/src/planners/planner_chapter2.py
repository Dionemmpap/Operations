import numpy as np
import time
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import heapq
from utils.obstacles import merge_intersecting_obstacles, get_obstacles
from utils.geometry import is_path_blocked, is_point_on_boundary
from utils.visualization import PlannerVisualizer  # Import the visualizer

class RecedingHorizonController:
    """
    Implements the receding horizon control for trajectory design as described in the paper.
    """
    def __init__(self, map_boundary, obstacles, start_point, end_point, N=30, Ne=3, tau=0.2, umax=1.0, use_visualizer=True):
        self.map_boundary = map_boundary
        # self.obstacles = merge_intersecting_obstacles(obstacles)
        self.obstacles = obstacles
        self.start_point = np.array(start_point)
        self.end_point = np.array(end_point)
        self.N = N  # Planning horizon
        self.Ne = Ne  # Execution horizon
        self.tau = tau  # Time step
        self.umax = umax  # Max control input
        self.BOUNDARY_MARGIN = 0.1  # Margin for boundary constraints

        #for sensitivity analysis:
        self.penalty_values = []
        self.arrival_time = None
        self.milp_times = []
        self.total_computation_time = 0.0
        self.distance_history = []



        # Build visibility graph and compute shortest paths with Dijkstra's
        self.graph, self.points = self._build_visibility_graph()
        self.cost_to_go = self._dijkstra()
        
        self.trajectory = [self.start_point]
        self.distance_covered = 0.0  # Track total distance covered
        self.distance_history.append(self.distance_covered)

        
        # Initialize visualizer if requested
        self.use_visualizer = use_visualizer
        if self.use_visualizer:
            self.visualizer = PlannerVisualizer(width=800, height=800)
            # Calculate bounds to ensure everything is visible
            self.visualizer.set_world_bounds(self.obstacles, self.start_point, self.end_point)
            # Add initial visualization frame
            self._update_visualization(self.start_point, [self.start_point], None)
            
    def _build_visibility_graph(self):
        """
        Builds a graph of all obstacle vertices and the start/end points.
        Edges exist only if the path between two points is clear of obstacles.
        """
        points = [self.end_point]
        for obs in self.obstacles:
            points.extend(obs)
        
        # Using a tuple for dictionary keys
        graph = {tuple(p): {} for p in points}

        for i, p1 in enumerate(points):
            for j, p2 in enumerate(points):
                if i >= j: continue
                # if not is_path_blocked(p1, p2, self.obstacles):
                dist = np.linalg.norm(np.array(p1) - np.array(p2))
                graph[tuple(p1)][tuple(p2)] = dist
                graph[tuple(p2)][tuple(p1)] = dist
        
        return graph, [np.array(p) for p in points]

    def _dijkstra(self):
        """
        Computes the shortest path from the end_point to all other nodes in the graph.
        This represents the "cost-to-go" from any node.
        """
        distances = {tuple(p): float('inf') for p in self.points}
        distances[tuple(self.end_point)] = 0
        pq = [(0, tuple(self.end_point))]  # (distance, node)

        while pq:
            dist, current_node = heapq.heappop(pq)
            if dist > distances[current_node]:
                continue
            for neighbor, weight in self.graph[current_node].items():
                if distances[current_node] + weight < distances[neighbor]:
                    distances[neighbor] = distances[current_node] + weight
                    heapq.heappush(pq, (distances[neighbor], neighbor))
        return distances

    def plan_and_execute(self):
        """
        Main loop for the receding horizon control. It iteratively plans a
        trajectory segment (planning horizon) and executes the first part of it (execution horizon).
        """
        current_pos = self.start_point
        max_steps = 100
        
        for step in range(max_steps):
            if np.linalg.norm(current_pos - self.end_point) < 2:
                self.arrival_time = len(self.trajectory) - 1  # Time steps taken to reach goal
                print("Goal reached!")
                self.trajectory.append(self.end_point)
                # Final visualization update
                if self.use_visualizer:
                    self._update_visualization(current_pos, self.trajectory[-50:], None)
                break

            planned_path = self._solve_milp(current_pos)
            if planned_path is None:
                print("Failed to find a path.")
                break

            # Update visualization with the planned path
            if self.use_visualizer:
                # Estimate a heading based on the trajectory direction
                heading = 0
                if len(self.trajectory) > 1:
                    direction = self.trajectory[-1] - self.trajectory[-2]
                    heading = np.arctan2(direction[1], direction[0])
                
                self._update_visualization(current_pos, self.trajectory[-50:], planned_path, heading)

            # Execute the first Ne steps
            next_pos = planned_path[self.Ne]
            
            # Update distance covered
            segment_distance = np.linalg.norm(next_pos - current_pos)
            self.distance_covered += segment_distance
            
            self.trajectory.extend(planned_path[1:self.Ne + 1])
            current_pos = next_pos
            print(f"Step {step+1}: Moved to {np.round(current_pos,2)}, Distance covered: {self.distance_covered:.2f}")

        if step == max_steps - 1:
            print("Max steps reached, terminating.")

        self.total_computation_time = sum(self.milp_times)
            
        # Close the visualizer when done
        if self.use_visualizer:
            self.visualizer.close()

    def _update_visualization(self, current_pos, actual_trajectory, predicted_trajectory, heading=0):
        """Update the visualization with current state"""
        vehicle_pos = (current_pos[0], current_pos[1], heading)
        
        # Convert trajectory arrays to list of tuples for visualizer
        actual_traj_viz = [(pos[0], pos[1]) for pos in actual_trajectory]
        pred_traj_viz = None
        if predicted_trajectory is not None:
            pred_traj_viz = [(pos[0], pos[1]) for pos in predicted_trajectory]
        
        # For visualization: use polygon (square) obstacles directly
        viz_obstacles = self.obstacles.copy()  # Use actual polygon obstacles
        
        # Add endpoint as a special obstacle for visualization
        viz_obstacles.append((self.end_point[0], self.end_point[1], 1, (0, 255, 0)))  # Green target
        
        # Create debug info dictionary
        distance_to_goal = np.linalg.norm(current_pos - self.end_point)
        debug_info = {
            "Distance covered": f"{self.distance_covered:.2f}",
            "Distance to goal": f"{distance_to_goal:.2f}",
            "Steps taken": f"{len(self.trajectory) - 1}",
            "Position": f"({current_pos[0]:.2f}, {current_pos[1]:.2f})"
        }
        
        # Debug print
        print(f"Vehicle at {current_pos}, distance to goal: {distance_to_goal:.2f}, total distance: {self.distance_covered:.2f}")
        
        # Update the visualization
        self.visualizer.update(vehicle_pos, viz_obstacles, actual_traj_viz, pred_traj_viz, 
                               map_boundary=self.map_boundary, debug_info=debug_info)

    def _solve_milp(self, start_pos):
        """
        Solves the MILP to find the next trajectory segment using an optimization,
        with interpolated visibility constraints and terminal cost-to-go.
        """
        model = gp.Model("receding_horizon")
        model.setParam('OutputFlag', 0)

        # --- Variables ---
        x = model.addVars(self.N + 1, 2, lb=-GRB.INFINITY, name='x')
        u = model.addVars(self.N, 2, lb=-GRB.INFINITY, name='u')

        # --- Initial Position ---
        model.addConstr(x[0, 0] == start_pos[0])
        model.addConstr(x[0, 1] == start_pos[1])

        # --- Dynamics: x[t+1] = x[t] + tau * u[t] ---
        model.addConstrs((x[t+1, i] == x[t, i] + self.tau * u[t, i]
                        for t in range(self.N) for i in range(2)), name="dynamics")

        # --- Control Constraints (bounded 2-norm) ---
        for t in range(self.N):
            model.addConstr(u[t, 0]*u[t, 0] + u[t, 1]*u[t, 1] <= self.umax*self.umax)

        # --- Obstacle Avoidance for Each State ---
        for t in range(1, self.N + 1):
            for obs in self.obstacles:
                b = model.addVars(4, vtype=GRB.BINARY)
                M = 1000
                xmin, ymin = np.min(obs, axis=0)
                xmax, ymax = np.max(obs, axis=0)
                model.addConstr(x[t, 0] <= xmin - 1e-2 + M * b[0])
                model.addConstr(x[t, 0] >= xmax + 1e-2 - M * b[1])
                model.addConstr(x[t, 1] <= ymin - 1e-2 + M * b[2])
                model.addConstr(x[t, 1] >= ymax + 1e-2 - M * b[3])
                model.addConstr(gp.quicksum(b) <= 3)

        # --- Map Boundary Constraints ---
        # Find the bounding box of the map boundary
        boundary_points = np.array(self.map_boundary)
        map_xmin, map_ymin = np.min(boundary_points, axis=0)
        map_xmax, map_ymax = np.max(boundary_points, axis=0)

        # Add constraints to keep all points within the map boundaries
        for t in range(1, self.N + 1):
            model.addConstr(x[t, 0] >= map_xmin + self.BOUNDARY_MARGIN)  # Small margin for safety
            model.addConstr(x[t, 0] <= map_xmax - self.BOUNDARY_MARGIN)
            model.addConstr(x[t, 1] >= map_ymin + self.BOUNDARY_MARGIN)
            model.addConstr(x[t, 1] <= map_ymax - self.BOUNDARY_MARGIN)

        # --- Terminal Cost-to-go with Interpolated Visibility ---
        vis_nodes = self.points  # All graph nodes are possible x_vis
        
        b_vis = model.addVars(len(vis_nodes), vtype=GRB.BINARY, name='b_vis')
        model.addConstr(gp.quicksum(b_vis[j] for j in range(len(vis_nodes))) == 1)

        # x_vis position and cost
        x_vis = model.addVars(2, lb=-GRB.INFINITY, name='x_vis')
        for i in range(2):
            model.addConstr(x_vis[i] == gp.quicksum(b_vis[j] * vis_nodes[j][i]
                                                    for j in range(len(vis_nodes))))
        c_vis = model.addVar(name='c_vis')
        model.addConstr(c_vis == gp.quicksum(b_vis[j] * self.cost_to_go[tuple(vis_nodes[j])]
                                            for j in range(len(vis_nodes))))

        # Interpolated visibility constraints between x[N] and x_vis
        T_interp = np.linspace(0.05, 0.95, 10)
        M = 1000
        for tau in T_interp:
            x_interp = model.addVars(2, name=f"x_interp_{tau:.2f}")
            for i in range(2):
                model.addConstr(
                    x_interp[i] == x[self.N, i] + tau * (x_vis[i] - x[self.N, i])
                )

            for obs in self.obstacles:
                xmin, ymin = np.min(obs, axis=0)
                xmax, ymax = np.max(obs, axis=0)
                b = model.addVars(4, vtype=GRB.BINARY)
                model.addConstr(x_interp[0] <= xmin - 1e-2 + M * b[0])
                model.addConstr(x_interp[0] >= xmax + 1e-2 - M * b[1])
                model.addConstr(x_interp[1] <= ymin - 1e-2 + M * b[2])
                model.addConstr(x_interp[1] >= ymax + 1e-2 - M * b[3])
                model.addConstr(gp.quicksum(b) <= 3)

        # Final Objective: L2(x[N] - x_vis) + c_vis
        dist_sq = model.addVar(name="dist_sq")
        model.addConstr(dist_sq == (x[self.N, 0] - x_vis[0]) * (x[self.N, 0] - x_vis[0]) +
                                    (x[self.N, 1] - x_vis[1]) * (x[self.N, 1] - x_vis[1]))
        effort = gp.quicksum(u[t, 0] * u[t, 0] + u[t, 1] * u[t, 1] for t in range(self.N))
        model.setObjective(dist_sq + c_vis + effort*0.01, GRB.MINIMIZE)

        # --- Solve the Model ---
        model.optimize()

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        self.milp_times.append(end_time - start_time)

        if model.status == GRB.OPTIMAL:

            # Extract terminal penalty components
            penalty = dist_sq.X + c_vis.X + effort.getValue() * 0.01
            self.penalty_values.append(penalty)

            trajectory = np.array([[x[t, 0].X, x[t, 1].X] for t in range(self.N + 1)])
            return trajectory
        
    
        else:
            print("Final optimization failed.")
            return None

    
    def plot_results(self):
        """
        Visualizes the final trajectory, obstacles, and map boundaries.
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot boundaries
        boundary_x, boundary_y = zip(*self.map_boundary + [self.map_boundary[0]])
        ax.plot(boundary_x, boundary_y, 'k-', linewidth=2, label='Map Boundary')

        # Plot obstacles
        for obs in self.obstacles:
            ox, oy = zip(*obs + [obs[0]])
            ax.fill(ox, oy, 'gray', edgecolor='black')

        # Plot start and end points
        ax.plot(self.start_point[0], self.start_point[1], 'go', markersize=10, label="Start")
        ax.plot(self.end_point[0], self.end_point[1], 'ro', markersize=10, label="End")
        
        # Plot trajectory
        if self.trajectory:
            traj_x, traj_y = zip(*self.trajectory)
            ax.plot(traj_x, traj_y, 'b-o', markersize=4, label="Planned Trajectory")

        ax.set_aspect('equal', 'box')
        
        # Move legend outside the plot
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Adjust layout to make room for the legend
        plt.tight_layout()
        plt.subplots_adjust(right=0.8)  # Make room for legend on the right
        
        plt.grid(True)
        plt.title("Final Planned Trajectory")
        plt.show()
