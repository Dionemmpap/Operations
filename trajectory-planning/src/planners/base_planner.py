import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import heapq
from utils.obstacles import merge_intersecting_obstacles,get_obstacles
from utils.geometry import is_path_blocked,is_point_on_boundary


class TrajectoryDesignBase():
    """Class to design a trajectory using receding horizon control."""
    def __init__(self, map_boundary, obstacles, end_point, start_point, tau):
        self.map_boundary = map_boundary
        self.obstacles = merge_intersecting_obstacles(obstacles)
        self.end_point = end_point
        self.start_point = start_point
        self.tau = tau
        self.graph, self.points = self.build_graph()
        self.distances = self.dijkstra()
        self.N = 10  # planning horizon
        self.Ne = 1  # execution steps per iteration
        self.umax = 1.0  # maximum step size

    def build_graph(self):
        """Creates a dictionary of points and their distances to other points to which the path is not blocked."""          
        points = []
        for obstacle in self.obstacles:
            for point in obstacle:
                # Only add points that are not on the boundary
                if not is_point_on_boundary(point, self.map_boundary):
                    points.append(point)
        
        # # Always include the end point
        # if not is_point_on_boundary(self.end_point, self.map_boundary):
        #     points.append(self.end_point)
        # else:
        #     # If end point is on boundary, we still need it
        #     points.append(self.end_point)
        points.append(self.end_point)

        graph = {}
        for i, point1 in enumerate(points):
            graph[tuple(point1)] = {}
            for j, point2 in enumerate(points):
                if i != j and not is_path_blocked(point1, point2, self.obstacles):
                    dist = np.linalg.norm(np.array(point1) - np.array(point2))
                    graph[tuple(point1)][tuple(point2)] = dist
        for point in points:
            if not is_path_blocked(point, self.end_point, self.obstacles):
                dist = np.linalg.norm(np.array(point) - np.array(self.end_point))
                graph[tuple(point)][tuple(self.end_point)] = dist
                graph[tuple(self.end_point)][tuple(point)] = dist

        return graph, points

    
    def dijkstra(self):
        """Dijkstra's algorithm to find the shortest path from endpoint to all other nodes."""
        edges = [(node, neighbor, weight) for node, neighbors in self.graph.items() for neighbor, weight in neighbors.items()]
        adj = {node: [] for node in self.graph}
        for node, neighbor, weight in edges:
            adj[node].append((neighbor, weight))
            adj[neighbor].append((node, weight))

        shortest = {} # Map vertex to its shortest distance from the endpoint
        minHeap = [[0, tuple(self.end_point)]]
        while minHeap:
            w1, n1 = heapq.heappop(minHeap)
            if n1 in shortest:
                continue
            shortest[n1] = w1
            for n2, w2 in adj[n1]:
                if n2 not in shortest:
                    heapq.heappush(minHeap, [w1 + w2, n2])

        return shortest

    def receding_horizon(self):
        current_position = self.start_point
        self.trajectory = []
        self.trajectory.append(current_position)  # Add starting point to trajectory
        
        # Add step counter to prevent infinite loops
        max_steps = 500
        step_counter = 0
        
        while not np.allclose(current_position, self.end_point, atol=1e-1):
            # Plan a trajectory from the current position to the endpoint
            next_position = self.plan_trajectory(current_position)
            print(f"Moving from {current_position} to {next_position}")
            
            # Store the trajectory
            self.trajectory.append(next_position)

            # Move to the next position
            current_position = next_position
            
            # Increment step counter and check for maximum steps
            step_counter += 1
            if step_counter > max_steps:
                print("Aborting: maximum steps reached.")
                break
                
            # Uncomment to see trajectory progress per iteration
            # self.plot(plt_traj=True)

    def plan_trajectory(self, current_position):
        N = self.N
        tau = self.tau

        model = gp.Model()
        model.setParam('OutputFlag', 0)

        # Variables for positions and control inputs
        x = model.addVars(N+1, 2, lb=-GRB.INFINITY, name='x')  # Positions x[0] to x[N]
        u = model.addVars(N, 2, lb=-self.umax, ub=self.umax, name='u')  # Inputs u[0] to u[N-1]

        # Initial condition
        model.addConstrs(x[0, i] == current_position[i] for i in range(2))

        # Dynamics
        for t in range(N):
            for i in range(2):
                model.addConstr(x[t+1, i] == x[t, i] + tau * u[t, i])

        # Basic obstacle avoidance for all x[t]
        for t in range(N+1):
            for obs in self.obstacles:
                xmin, xmax = np.min([p[0] for p in obs]), np.max([p[0] for p in obs])
                ymin, ymax = np.min([p[1] for p in obs]), np.max([p[1] for p in obs])
                b = model.addVars(4, vtype=GRB.BINARY)
                M = 1000
                model.addConstr(x[t, 0] <= xmin - 1e-2 + M * b[0])
                model.addConstr(x[t, 0] >= xmax + 1e-2 - M * b[1])
                model.addConstr(x[t, 1] <= ymin - 1e-2 + M * b[2])
                model.addConstr(x[t, 1] >= ymax + 1e-2 - M * b[3])
                model.addConstr(gp.quicksum(b[i] for i in range(4)) <= 3)

        # First solve: no terminal penalty yet
        model.setObjective(gp.quicksum(u[t, 0]*u[t, 0] + u[t, 1]*u[t, 1] for t in range(N)), GRB.MINIMIZE)
        model.optimize()

        # Get terminal point after first solve
        xN = (x[N, 0].X, x[N, 1].X)

        # Get visible cost nodes
        vis_nodes = [node for node in self.distances.keys()
                    if not is_path_blocked(xN, node, self.obstacles)]
        if not vis_nodes:
            print("No visible nodes from terminal point — fallback triggered.")
            return np.array(xN)

        # Add visibility selection binary
        b_vis = model.addVars(len(vis_nodes), vtype=GRB.BINARY, name='b_vis')
        model.addConstr(gp.quicksum(b_vis[j] for j in range(len(vis_nodes))) == 1)

        # Define x_vis and c_vis
        x_vis = model.addVars(2, name='x_vis')
        for i in range(2):
            model.addConstr(x_vis[i] == gp.quicksum(b_vis[j] * vis_nodes[j][i] for j in range(len(vis_nodes))))
        c_vis = model.addVar(name='c_vis')
        model.addConstr(c_vis == gp.quicksum(b_vis[j] * self.distances[tuple(vis_nodes[j])] for j in range(len(vis_nodes))))

        # Add visibility constraints between x[N] and x_vis
        T_interp = np.linspace(0.05, 0.95, 15)
        for t_frac in T_interp:
            x_interp = model.addVars(2, name=f'interp_{t_frac:.2f}')
            for i in range(2):
                model.addConstr(x_interp[i] == x[N, i] + t_frac * (x_vis[i] - x[N, i]))
            for obs in self.obstacles:
                xmin, xmax = np.min([p[0] for p in obs]), np.max([p[0] for p in obs])
                ymin, ymax = np.min([p[1] for p in obs]), np.max([p[1] for p in obs])
                b = model.addVars(4, vtype=GRB.BINARY)
                M = 1000
                model.addConstr(x_interp[0] <= xmin - 1e-2 + M * b[0])
                model.addConstr(x_interp[0] >= xmax + 1e-2 - M * b[1])
                model.addConstr(x_interp[1] <= ymin - 1e-2 + M * b[2])
                model.addConstr(x_interp[1] >= ymax + 1e-2 - M * b[3])
                model.addConstr(gp.quicksum(b[i] for i in range(4)) <= 3)

        # Define L1 norm distance from x[N] to x_vis
        diff = model.addVars(2, name='diff')
        abs_diff = model.addVars(2, name='abs_diff')
        for i in range(2):
            model.addConstr(diff[i] == x_vis[i] - x[N, i])
            model.addGenConstrAbs(abs_diff[i], diff[i])

        # Set objective: L1(x[N] to x_vis) + c_vis
        model.setObjective(gp.quicksum(abs_diff[i] for i in range(2)) + c_vis, GRB.MINIMIZE)

        # Re-solve with terminal cost
        model.optimize()
        print(f"  x[N]    = {xN}")
        chosen_index = [j for j in range(len(vis_nodes)) if b_vis[j].X > 0.5][0]
        print(f"  x_vis   = {vis_nodes[chosen_index]}")
        print(f"  cost-to-go = {self.distances[tuple(vis_nodes[chosen_index])]:.2f}")
        

        if model.status != GRB.OPTIMAL:
            print("Warning: MILP model not optimal.")
            return np.array(xN)

        # Return the next point
        next_point = np.array([x[self.Ne, 0].X, x[self.Ne, 1].X])
        print(f"  step size: {np.linalg.norm(next_point - current_position):.3f}")
        return next_point




    def plot(self,plt_traj=False):
        """Plot the map, obstacles, and network."""
        #visualize_map(self.map_boundary, self.obstacles, self.graph, self.end_point)
        #add the trajectory to the plot
        fig, ax = plt.subplots()
        boundary_x, boundary_y = zip(*self.map_boundary + [self.map_boundary[0]])
        ax.plot(boundary_x, boundary_y, color='black', label='Boundary')
        for i, obstacle in enumerate(self.obstacles):
            obstacle_x, obstacle_y = zip(*obstacle + [obstacle[0]])
            ax.plot(obstacle_x, obstacle_y, label=f'Obstacle {i+1}', linestyle='--')

        """Uncomment this if you'd like to see the network"""
        #for node, neighbors in self.graph.items():
        #    for neighbor in neighbors:
        #        ax.plot(
        #            [node[0], neighbor[0]], [node[1], neighbor[1]], color='blue', alpha=0.5
        #        )
        
        ax.scatter(*self.end_point, color='red', label='Endpoint')
        ax.scatter(*self.start_point, color='green', label='Startpoint')
        if plt_traj:
            for i in range(len(self.trajectory)-1):
                ax.plot([self.trajectory[i][0], self.trajectory[i+1][0]], [self.trajectory[i][1], self.trajectory[i+1][1]], color='green')
        ax.set_aspect('equal')
        plt.legend()
        plt.title("Map with Obstacles and Network")
        plt.show()