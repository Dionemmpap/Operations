""" Receding Horizon Control for Trajectory Design """
import numpy as np
import heapq
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
from shapely.geometry import Point, Polygon, MultiPolygon, LineString


#Helper Functions
#def equal_points(p1, p2):
#    """Check if two points are equal."""
#    return np.allclose(p1, p2, atol=1e-1)

def point_inside_obstacle(point, obstacles):
    """
    Check if a point is inside any obstacle.
    
    Args:
        point (list or tuple): The [x, y] coordinates of the point.
        obstacles (list of lists): List of obstacles, where each obstacle is a list of [x, y] vertices.
    
    Returns:
        bool: True if the point is inside any obstacle, False otherwise.
    """
    point_geom = Point(point)
    for obstacle in obstacles:
        obstacle_polygon = Polygon(obstacle)
        if obstacle_polygon.contains(point_geom):
            return True
    return False


#def path_is_diagonal_of_obstacle(p1, p2, obstacles):
#    """Check if a straight line between two points is not the diagnal a the obstacle."""
#    def check_diagonal(p1, p2, obstacle, n):
#        return equal_points(p1, obstacle[n%4]) and equal_points(p2, obstacle[(n+2)%4])
#    
#    for obstacle in obstacles:
#        for i in range(4):
#            if check_diagonal(p1, p2, obstacle, i):
#                return False
#    return True

#def lines_intersect(p1, p2, q1, q2):
#    """Check if line segments p1-p2 and q1-q2 intersect."""
#    def ccw(a, b, c):
#        """Check if points a, b, c are listed in a counterclockwise order."""
#        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
#
#    if equal_points(p1, q1) or equal_points(p1, q2) or equal_points(p2, q1) or equal_points(p2, q2):
#        return False
#    
#    # Two line segments intersect if and only if they straddle each other
#    return ccw(p1, q1, q2) != ccw(p2, q1, q2) and ccw(p1, p2, q1) != ccw(p1, p2, q2)

def is_path_blocked(point1, point2, obstacles):
    """Check if the straight line between two points intersects or lies within any obstacle."""
    line = LineString([point1, point2])
    for obstacle in obstacles:
        polygon = Polygon(obstacle)
        if line.crosses(polygon) or line.within(polygon):
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

def random_shift_point(point, shift_range=0.1):
    """Shift a point by a random amount within a range."""
    x_shift = np.random.uniform(-shift_range, shift_range)
    y_shift = np.random.uniform(-shift_range, shift_range)
    return [point[0] + x_shift, point[1] + y_shift]

def random_shift_all(obstacles, shift_range=0.1):
    """Shift the vertices of obstacles by a random amount within a range."""
    shifted_obstacles = []
    for obstacle in obstacles:
        shifted_obstacle = []
        for vertex in obstacle:
            shifted_obstacle.append(random_shift_point(vertex, shift_range))
        shifted_obstacles.append(shifted_obstacle)
    return shifted_obstacles

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
            points.extend(obstacle)
        points.append(self.end_point)

        graph = {}
        for i, point1 in enumerate(points):
            graph[tuple(point1)] = {}
            for j, point2 in enumerate(points):
                if i != j and not is_path_blocked(point1, point2, self.obstacles): #and path_is_diagonal_of_obstacle(point1, point2, self.obstacles):
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
        self.trajectory = np.empty((0,2), float)
        while not np.allclose(current_position, self.end_point, atol=1e-1):
            # Plan a trajectory from the current position to the endpoint
            next_position = self.plan_trajectory(current_position)
            print(f"Moving from {current_position} to {next_position}")
            #store the trajectory
            self.trajectory = np.append(self.trajectory, [next_position], axis=0)

            # Move to the next position
            current_position = next_position
            #self.plot(plt_traj=True)

    def plan_trajectory(self, current_position):
        """Plan a trajectory from the current position to the endpoint."""
        # Find visible nodes from the current position
        visible_nodes = []
        for node in self.points:
            if not is_path_blocked(current_position, node, self.obstacles):
                visible_nodes.append(node)

        # Filter out trajectories ending inside obstacles
        valid_nodes = [
            node for node in visible_nodes #if not point_inside_obstacle(node, self.obstacles)
        ]

        #if not valid_nodes:
            #raise ValueError("No valid trajectories available due to obstacles.")

        # Solve optimization problem to find the best node
        objective = gp.Model()
        objective.setParam('OutputFlag', 0)
        x = objective.addVars(len(valid_nodes), vtype=GRB.BINARY, name='x')
        objective.setObjective(
            sum(
                x[j] * (
                    np.linalg.norm(np.array(current_position) - np.array(valid_nodes[j]))
                    + self.distances[tuple(valid_nodes[j])]
                ) for j in range(len(valid_nodes))
            ), GRB.MINIMIZE
        )
        objective.addConstr(sum(x[j] for j in range(len(valid_nodes))) == 1)
        objective.optimize()

        # Calculate the proposed next position
        direction = np.array(valid_nodes[np.argmax([x[j].x for j in range(len(valid_nodes))])]) - np.array(current_position)
        direction = direction / np.linalg.norm(direction)  # Normalize the direction vector
        proposed_point = np.array(current_position) + self.tau * direction

        # Validate the proposed point
        #if point_inside_obstacle(proposed_point, self.obstacles):
            #raise ValueError(f"Proposed trajectory point {proposed_point} is inside an obstacle.")

        return proposed_point
    
    
    def calc_trajectory_length(self):
        """Calculate the length of the trajectory."""
        length = 0
        for i in range(len(self.trajectory)-1):
            length += np.linalg.norm(np.array(self.trajectory[i]) - np.array(self.trajectory[i+1]))
        return length


    def plot(self,plt_traj=False, show_network=False, extra_trajectory_list=None):
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
        if show_network:
            for node, neighbors in self.graph.items():
                for neighbor in neighbors:
                    ax.plot(
                        [node[0], neighbor[0]], [node[1], neighbor[1]], color='blue', alpha=0.5
                    )
        
        ax.scatter(*self.end_point, color='red', label='Endpoint')
        ax.scatter(*self.start_point, color='green', label='Startpoint')
        if plt_traj:
            ax.plot(self.trajectory[:,0], self.trajectory[:,1], color="black", label="Trajectory")
        
        if extra_trajectory_list:
            colors = ['orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta', 'yellow', 'green']
            for j, extra_trajectory in enumerate(extra_trajectory_list):
                ax.plot(extra_trajectory[:,0], extra_trajectory[:,1], linestyle = ":", color=colors[j%len(colors)], label=f"Trajectory {j+1}")
        
        ax.set_aspect('equal')
        plt.annotate(f"Trajectory Length: {self.calc_trajectory_length():.2f}", self.start_point)
        plt.legend()
        plt.title("Map with Obstacles and Network")
        plt.show()
        return ax


def run_tau(number_of_obstacles, tau_list):
    # Define map boundary and obstacles
    map_boundary = [[0, 0], [10, 0], [10, 10], [0, 10]]
    obstacles = get_obstacles(map_boundary, number_of_obstacles)
    end_point = [9, 9]
    trajectory_list = []

    visualize_map(map_boundary, obstacles, {}, end_point)

    for i, tau in enumerate(tau_list):
        td = TrajectoryDesign(map_boundary, obstacles, end_point, [0, 0], tau)

        td.receding_horizon()

        if i == 0:
            td.plot(plt_traj=True)
            org_td = td
        else:
            trajectory_list.append(td.trajectory)
    
        print(f"Trajectory Length: {td.calc_trajectory_length():.2f}")
    org_td.plot(plt_traj=True, extra_trajectory_list=trajectory_list)



def run_with_shifts(number_of_obstacles, tau, shift_range, num_shifts):
    # Define map boundary and obstacles
    map_boundary = [[0, 0], [10, 0], [10, 10], [0, 10]]
    org_obstacles = get_obstacles(map_boundary, number_of_obstacles)
    org_end_point = [9, 9]
    org_start_point = [0, 0]
    visualize_map(map_boundary, org_obstacles, {}, org_end_point)
    trajectory_list = []

    for i in range(num_shifts):
        if i == 0:
            obstacles = org_obstacles
            end_point = org_end_point
            start_point = org_start_point
        else:
            obstacles = random_shift_all(org_obstacles, shift_range)
            end_point = random_shift_point(org_end_point, shift_range)
            start_point = random_shift_point(org_start_point, shift_range)


        td = TrajectoryDesign(map_boundary, obstacles, end_point, start_point, tau)

        td.receding_horizon()

        if i == 0:
            td.plot(plt_traj=True)
            org_td = td
        else:
            trajectory_list.append(td.trajectory)
    
        print(f"Trajectory Length: {td.calc_trajectory_length():.2f}")
    org_td.plot(plt_traj=True, extra_trajectory_list=trajectory_list)

def run():
    # Define map boundary and obstacles
    map_boundary = [[0, 0], [10, 0], [10, 10], [0, 10]]
    obstacles = get_obstacles(map_boundary, 5)
    #If you'd like a custom obstacle, you can add it here
    #obstacles.append([[5, 5], [6, 5], [6, 6], [5, 6]])
    end_point = [9.9, 9.9]

    visualize_map(map_boundary, obstacles, {}, end_point)

    td = TrajectoryDesign(map_boundary, obstacles, end_point, [0, 0], 0.1)

    td.receding_horizon()
    td.plot(plt_traj=True)
    
    # # Print distances for each obstacle corner to the endpoint
    print("Distances from obstacle corners to endpoint:")
    for obstacle in td.obstacles:
        for corner in obstacle:
            print(f"Corner {corner} -> Endpoint: {td.distances[tuple(corner)]:.2f}")

    print(f"Trajectory Length: {td.calc_trajectory_length():.2f}")
    

def main():
    #run()
    # run_with_shifts(5, 0.1, 0.2, 5)
    run_tau(5, [0.05, 0.075, 0.1, 0.125, 0.15, 0.2])
   
    

if __name__ == "__main__":
    main()
