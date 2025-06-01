""" Receding Horizon Control for Trajectory Design """
import numpy as np
import heapq
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
from shapely.geometry import Point, Polygon, MultiPolygon, LineString


#Helper Functions
def is_path_blocked(point1, point2, obstacles):
    """Check if the straight line between two points intersects or lies within any obstacle."""
    line = LineString([point1, point2])
    for obstacle in obstacles:
        polygon = Polygon(obstacle)
        if line.crosses(polygon) or line.within(polygon):
            return True
    return False

def is_point_on_boundary(point, map_boundary):
    """Check if a point lies on the map boundary."""
    # Check if point matches any boundary point
    if point in map_boundary:
        return True
    
    # Check if point lies on any boundary edge
    for i in range(len(map_boundary)):
        p1 = map_boundary[i]
        p2 = map_boundary[(i + 1) % len(map_boundary)]
        
        # Create line segment
        line = LineString([p1, p2])
        
        # Check if point lies on the line segment
        if line.distance(Point(point)) < 1e-6:
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



def get_obstacles(map_boundary, num_obstacles):
    obstacles = []
    for _ in range(num_obstacles):
        x1 = np.random.uniform(0.15*map_boundary[1][0], 0.75*map_boundary[1][0])
        y1 = np.random.uniform(0.15*map_boundary[2][1], 0.75*map_boundary[2][1])
        width = np.random.uniform(0.1*map_boundary[1][0], 0.25*map_boundary[1][0])
        height = np.random.uniform(0.1*map_boundary[2][1], 0.25*map_boundary[2][1])
        x2 = x1 + width
        y2 = y1
        x3 = x2
        y3 = y1 + height
        x4 = x1
        y4 = y3
        obstacles.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])


    return obstacles

def merge_intersecting_obstacles(obstacles):
        """Merge intersecting obstacles into a single larger obstacle."""
        # Convert obstacles to shapely Polygons
        polygons = [Polygon(obstacle) for obstacle in obstacles]

        # Merge all polygons using buffer(0)
        merged = MultiPolygon(polygons).buffer(0)

        # Check if merged result is a single Polygon
        if isinstance(merged, Polygon):
            return [list(merged.exterior.coords[:-1])]

        # Check if merged result is a MultiPolygon
        elif isinstance(merged, MultiPolygon):
            return [list(poly.exterior.coords[:-1]) for poly in merged.geoms]

        # If no merging occurs (fallback)
        return obstacles



class TrajectoryDesign():
    """Class to design a trajectory using receding horizon control."""
    def __init__(self, map_boundary, obstacles, end_point, start_point, tau):
        self.map_boundary = map_boundary
        #self.obstacles = obstacles
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
            #self.plot(plt_traj=True)

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
        #plt.legend()
        plt.title("Map with Obstacles and Network")
        plt.show()


    
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

    td = TrajectoryDesign(map_boundary, obstacles, end_point, [0, 5], 0.1)

    td.receding_horizon()
    td.plot(plt_traj=True)
    
    # # Print distances for each obstacle corner to the endpoint
    print("Distances from obstacle corners to endpoint:")
    for obstacle in td.obstacles:
        for corner in obstacle:
            print(f"Corner {corner} -> Endpoint: {td.distances[tuple(corner)]:.2f}")

if __name__ == "__main__":
    main()
