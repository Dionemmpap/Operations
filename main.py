""" Receding Horizon Control for Trajectory Design """
import numpy as np
import heapq
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB




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


# def lines_intersect(p1, p2, q1, q2):
#     """Check if line segments p1-p2 and q1-q2 intersect using Gurobi."""
#     model = gp.Model()
#     model.setParam('OutputFlag', 0)

#     # Define variables for intersection points
#     t = model.addVar(lb=0, ub=1, name="t")
#     u = model.addVar(lb=0, ub=1, name="u")

#     # Line equations
#     model.addConstr((q1[0] - p1[0]) * t == (p2[0] - p1[0]) * u)
#     model.addConstr((q1[1] - p1[1]) * t == (p2[1] - p1[1]) * u)

#     # Solve the model
#     model.optimize()
#     if model.status == GRB.OPTIMAL:
#         return True

#     return False



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
        pq = [(0, tuple(self.end_point))]
        distances = {node: float('inf') for node in self.graph}
        distances[tuple(self.end_point)] = 0

        while pq:
            current_distance, current_node = heapq.heappop(pq)

            if current_distance > distances[current_node]:
                continue

            for neighbor, weight in self.graph[current_node].items():
                distance = current_distance + weight
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heapq.heappush(pq, (distance, neighbor))
        
        return distances
    
    def receding_horizon(self):
        pass

    def plot(self):
        """Plot the map, obstacles, and network."""
        visualize_map(self.map_boundary, self.obstacles, self.graph, self.end_point)

    
def main():
    # Define map boundary and obstacles
    map_boundary = [[0, 0], [10, 0], [10, 10], [0, 10]]
    obstacles = [
        [[2, 2], [4, 2], [4, 4], [2, 4]],  # Obstacle 1
        [[6, 6], [8, 6], [8, 8], [6, 8]],  # Obstacle 2
    ]
    end_point = [9, 9]

    td = TrajectoryDesign(map_boundary, obstacles, end_point, [1, 1], 0.1)

    td.plot()
    print(td.graph)
    
    # # Print distances for each obstacle corner to the endpoint
    print("Distances from obstacle corners to endpoint:")
    for obstacle in td.obstacles:
        for corner in obstacle:
            print(f"Corner {corner} -> Endpoint: {td.distances[tuple(corner)]:.2f}")

    

if __name__ == "__main__":
    main()
