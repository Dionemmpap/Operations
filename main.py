""" Receding Horizon Control for Trajectory Design """
import numpy as np
import heapq
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB


#Helper Functions
def lines_intersect(p1, p2, q1, q2):
    """Check if line segments p1-p2 and q1-q2 intersect."""
    def ccw(a, b, c):
        """Check if points a, b, c are listed in a counterclockwise order."""
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

    # Two line segments intersect if and only if they straddle each other
    return ccw(p1, q1, q2) != ccw(p2, q1, q2) and ccw(p1, p2, q1) != ccw(p1, p2, q2)

def is_path_blocked(point1, point2, obstacles):
    """Check if a straight line between two points intersects any obstacles."""
    for obstacle in obstacles:
        for i in range(len(obstacle)):
            p1 = obstacle[i]
            p2 = obstacle[(i + 1) % len(obstacle)]
            if lines_intersect(point1, point2, p1, p2):
                print(f"Blocked: {point1} -> {point2} by {p1}-{p2}")
                return True
    return False

def visualize_map(map_boundary, obstacles, graph, end_point):
    """Visualize the map, obstacles, and network."""
    fig, ax = plt.subplots()

    # Plot map boundary
    boundary_x, boundary_y = zip(*map_boundary + [map_boundary[0]])
    ax.plot(boundary_x, boundary_y, color='black', label='Boundary')

    # Plot obstacles
    for i, obstacle in enumerate(obstacles):
        obstacle_x, obstacle_y = zip(*obstacle + [obstacle[0]])
        ax.plot(obstacle_x, obstacle_y, label=f'Obstacle {i+1}', linestyle='--')

    # Plot graph edges
    for node, neighbors in graph.items():
        for neighbor in neighbors:
            ax.plot(
                [node[0], neighbor[0]], [node[1], neighbor[1]], color='blue', alpha=0.5
            )

    # Plot end point
    ax.scatter(*end_point, color='red', label='Endpoint')

    ax.set_aspect('equal')
    plt.legend()
    plt.title("Map with Obstacles and Network")
    plt.show()


class TrajectoryDesign():
    """Class to design a trajectory using receding horizon control."""
    def __init__(self, map_boundary, obstacles, end_point, start_point, tau):
        self.map_boundary = map_boundary
        self.obstacles = obstacles
        self.end_point = end_point
        self.start_point = start_point
        self.tau = tau
        self.graph, self.points = self.build_graph()
        self.distances = self.dijkstra()

    def build_graph(self):
        """Creates a dictionary of points and their distances to other points to which the path is not blocked."""	
        points = []
        for obstacle in self.obstacles:
            points.extend(obstacle)
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
        while not np.allclose(current_position, self.end_point, atol=1e-6):
            # Plan a trajectory from the current position to the endpoint
            next_position = self.plan_trajectory(current_position)
            print(f"Moving from {current_position} to {next_position}")
            #store the trajectory
            self.trajectory.append(next_position)

            # Move to the next position
            current_position = next_position

    def plan_trajectory(self, current_position):
        """Plan a trajectory from the current position to the endpoint."""

        #Find visible nodes from the current position
        visible_nodes = []
        for node in self.points:
            if not is_path_blocked(current_position, node, self.obstacles):
                visible_nodes.append(node)
        #Find the node with the shortest distance to the endpoint: f(x(N)) = |x(N)-x_v,j| + c_j, where x(N) is the current position, x_v,j is the visible node, and c_j is the distance from the visible node to the endpoint
        objective = gp.Model()
        objective.setParam('OutputFlag', 0)
        x = objective.addVars(len(visible_nodes), vtype=GRB.BINARY, name='x')
        objective.setObjective(sum(x[j] * (np.linalg.norm(np.array(current_position) - np.array(visible_nodes[j])) + self.distances[tuple(visible_nodes[j])]) for j in range(len(visible_nodes))), GRB.MINIMIZE)
        objective.addConstr(sum(x[j] for j in range(len(visible_nodes))) == 1)
        objective.optimize()
        point = np.array(current_position) + self.tau * (np.array(visible_nodes[np.argmax([x[j].x for j in range(len(visible_nodes))])]) - np.array(current_position))
                                                         
        return point
            
    


    def plot(self,plt_traj=False):
        """Plot the map, obstacles, and network."""
        # visualize_map(self.map_boundary, self.obstacles, self.graph, self.end_point)
        #add the trajectory to the plot
        fig, ax = plt.subplots()
        boundary_x, boundary_y = zip(*self.map_boundary + [self.map_boundary[0]])
        ax.plot(boundary_x, boundary_y, color='black', label='Boundary')
        for i, obstacle in enumerate(self.obstacles):
            obstacle_x, obstacle_y = zip(*obstacle + [obstacle[0]])
            ax.plot(obstacle_x, obstacle_y, label=f'Obstacle {i+1}', linestyle='--')
        for node, neighbors in self.graph.items():
            for neighbor in neighbors:
                ax.plot(
                    [node[0], neighbor[0]], [node[1], neighbor[1]], color='blue', alpha=0.5
                )
        ax.scatter(*self.end_point, color='red', label='Endpoint')
        ax.scatter(*self.start_point, color='green', label='Startpoint')
        if plt_traj:
            for i in range(len(self.trajectory)-1):
                ax.plot([self.trajectory[i][0], self.trajectory[i+1][0]], [self.trajectory[i][1], self.trajectory[i+1][1]], color='green')
        ax.set_aspect('equal')
        plt.legend()
        plt.title("Map with Obstacles and Network")
        plt.show()


    
def main():
    # Define map boundary and obstacles
    map_boundary = [[0, 0], [10, 0], [10, 10], [0, 10]]
    obstacles = [
        [[2, 2], [4, 2], [4, 4], [2, 4]],  # Obstacle 1
        [[6, 6], [8, 6], [8, 8], [6, 8]],  # Obstacle 2
    ]
    end_point = [9, 9]

    td = TrajectoryDesign(map_boundary, obstacles, end_point, [1, 1], 0.1)

    td.receding_horizon()
    td.plot(plt_traj=False)
    
    # # Print distances for each obstacle corner to the endpoint
    print("Distances from obstacle corners to endpoint:")
    for obstacle in td.obstacles:
        for corner in obstacle:
            print(f"Corner {corner} -> Endpoint: {td.distances[tuple(corner)]:.2f}")
   
    

if __name__ == "__main__":
    main()
