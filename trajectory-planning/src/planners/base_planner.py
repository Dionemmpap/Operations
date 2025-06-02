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

    
    def build_graph(self):
        """Creates a dictionary of points and their distances to other points to which the path is not blocked."""	
        points = []
        for obstacle in self.obstacles:
            for point in obstacle:
                # Only add points that are not on the boundary
                if not is_point_on_boundary(point, self.map_boundary):
                    points.append(point)
        
        # Always include the end point
        if not is_point_on_boundary(self.end_point, self.map_boundary):
            points.append(self.end_point)
        else:
            # If end point is on boundary, we still need it
            points.append(self.end_point)

        graph = {}
        for i, point1 in enumerate(points):
            graph[tuple(point1)] = {}
            for j, point2 in enumerate(points):
                if i != j and not is_path_blocked(point1, point2, self.obstacles):
                    dist = np.linalg.norm(np.array(point1) - np.array(point2))
                    graph[tuple(point1)][tuple(point2)] = dist

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
        while not np.allclose(current_position, self.end_point, atol=1e-1):
            # Plan a trajectory from the current position to the endpoint
            next_position = self.plan_trajectory(current_position)
            print(f"Moving from {current_position} to {next_position}")
            #store the trajectory
            self.trajectory.append(next_position)

            # Move to the next position
            current_position = next_position

            #Uncomment to see trajectory progress per iteration
            # self.plot(plt_traj=True)

    def plan_trajectory(self, current_position):
        """Plan a trajectory from the current position to the endpoint."""
        # Find visible nodes from the current position
        visible_nodes = []
        for node in self.points:
            if not is_path_blocked(current_position, node, self.obstacles):
                visible_nodes.append(node)

        # Solve optimization problem to find the best node
        objective = gp.Model()
        objective.setParam('OutputFlag', 0)
        x = objective.addVars(len(visible_nodes), vtype=GRB.BINARY, name='x')
        objective.setObjective(
            sum(
                x[j] * (
                    np.linalg.norm(np.array(current_position) - np.array(visible_nodes[j]))
                    + self.distances[tuple(visible_nodes[j])]
                ) for j in range(len(visible_nodes))
            ), GRB.MINIMIZE
        )
        objective.addConstr(sum(x[j] for j in range(len(visible_nodes))) == 1)
        objective.optimize()

        # Calculate the proposed next position
        direction = np.array(visible_nodes[np.argmax([x[j].x for j in range(len(visible_nodes))])]) - np.array(current_position)
        direction = direction / np.linalg.norm(direction)  # Normalize the direction vector
        point = np.array(current_position) + self.tau * direction

        return point



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