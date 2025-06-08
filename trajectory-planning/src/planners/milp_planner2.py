import numpy as np
import heapq
from shapely.geometry import Point, Polygon, LineString
# Assuming your utility files are in the specified structure
from utils.geometry import (
    find_tangents_to_circle,
    is_valid_tangent,
    calculate_arc_length,
    is_path_obstructed,
    approximate_arc
)
from utils.obstacles import merge_intersecting_obstacles

# --- Refined node class for better alignment with the paper ---
class node(object):
    """
    A class to represent a node in the trajectory planning tree.
    Adjusted to align more closely with the paper's algorithm.
    """
    def __init__(self, pos, vel=None, cost_to_go=float('inf')):
        self.pos = pos
        # Velocity is unknown until a path to it is found
        self.vel = vel
        self.cost_to_go = cost_to_go
        self.circle_cw = None
        self.circle_ccw = None
        self.id = None
        # In the tree, each node has exactly one successor on the path to the goal
        self.successor = None

    def __repr__(self):
        # Format cost for readability
        cost_str = f"{self.cost_to_go:.2f}" if self.cost_to_go != float('inf') else "inf"
        return f"Node(id={self.id}, pos={self.pos}, cost={cost_str})"
        
    def calc_circle_centers(self, rho, vmax):
        # (This method from the previous answer remains unchanged)
        if self.vel is None or (self.vel[0] == 0 and self.vel[1] == 0):
            self.circle_cw = None
            self.circle_ccw = None
            return
        vx, vy = self.vel
        offset_cw_x = (rho / vmax) * vy
        offset_cw_y = (rho / vmax) * (-vx)
        self.circle_cw = (self.pos[0] + offset_cw_x, self.pos[1] + offset_cw_y)
        offset_ccw_x = (rho / vmax) * (-vy)
        offset_ccw_y = (rho / vmax) * vx
        self.circle_ccw = (self.pos[0] + offset_ccw_x, self.pos[1] + offset_ccw_y)


class MILPTrajectoryPlanner():

    def __init__(self, map_boundary, obstacles, goal_state, ini_state, v_max, turning_radius, Delta_T, N, N_exec):
        self.map_boundary = map_boundary
        # It's good practice to pre-process obstacles once
        self.obstacles = merge_intersecting_obstacles(obstacles)
        self.obstacle_polygons = [Polygon(obs) for obs in self.obstacles]
        
        self.end_point = goal_state[0]
        self.start_point = ini_state[0]
        self.v_max = v_max
        self.rho = turning_radius
        self.Delta_T = Delta_T
        self.N = N
        self.N_exec = N_exec
        self.model = None
        self.x_vars = None
        self.y_vars = None
        self.time_vars = None
        

    # Place this inside your MILPTrajectoryPlanner class
    def build_feasible_traj_tree(self, start_pos, goal_state):
        """
        Implements Algorithm 2: Make-Path-Tree to find a tree of 
        kinodynamically feasible paths to the goal.
        """
        import heapq
        
        # Convert obstacles to Shapely polygons for collision detection
        self.obstacle_polygons = [Polygon(obstacle) for obstacle in self.obstacles]
        
        # Extract valid vertices for nodes
        obstacle_vertices = self._get_valid_obstacle_vertices()
        print(f"Valid obstacle vertices: {len(obstacle_vertices)}")
        
        # Create nodes for the start and goal positions with IDs
        start_node = node(pos=start_pos, vel=(0, 0), cost_to_go=float('inf'))
        goal_node = node(pos=goal_state['pos'], vel=goal_state['vel'], cost_to_go=0)
        goal_node.id = 'goal'
        start_node.id = 'start'
        # Calculate turning circles for the goal node (critical for backward path finding)
        goal_node.calc_circle_centers(rho=self.rho, vmax=self.v_max)

        # Create a node for each obstacle vertex with IDs
        obstacle_nodes = []
        for i, vertex in enumerate(obstacle_vertices):
            obs_node = node(pos=vertex, vel=(0, 0), cost_to_go=float('inf'))
            obs_node.id = f'obs_{i}'
            obstacle_nodes.append(obs_node)
        
        # Combine all nodes into a single list
        all_nodes = [start_node, goal_node] + obstacle_nodes
        
        # Create a dictionary to easily access nodes by ID
        node_dict = {node.id: node for node in all_nodes}
        
        # Initialize the priority queue with all nodes
        # Format: (cost_to_go, counter, node_id) - counter helps with consistent ordering for equal costs
        counter = 0
        pq = []
        for n in all_nodes:
            heapq.heappush(pq, (n.cost_to_go, counter, n.id))
            counter += 1
        
        # Main Dijkstra-like loop - WORKS BACKWARDS FROM GOAL
        finalized_nodes = set()
        while pq:
            cost, _, current_id = heapq.heappop(pq)
            current_node = node_dict[current_id]
            
            # Skip if we've already processed this node or cost is infinite
            if current_id in finalized_nodes or cost == float('inf'):
                continue
            
            # Mark the current node as finalized
            finalized_nodes.add(current_id)
            print(f"Processing node {current_id} with cost {cost}")
            
            # Process all potential neighbors
            for neighbor_id, neighbor_node in node_dict.items():
                if neighbor_id in finalized_nodes:
                    continue

                # CRITICAL STEP: Find the kinodynamic path FROM neighbor (i) TO current (j)
                # The cost is NOT the straight-line distance.
                connection = self._find_connecting_segment(neighbor_node.pos, current_node)
                
                # If a valid, kinodynamic, obstacle-free path was found
                if connection:
                    path_length, initial_velocity, path_geometry = connection
                    
                    # The total cost to the goal from the neighbor is via the current node
                    new_cost_from_neighbor = current_node.cost_to_go + path_length
                    
                    if new_cost_from_neighbor < neighbor_node.cost_to_go:
                        # We found a better kinodynamic path! Update the neighbor.
                        neighbor_node.cost_to_go = new_cost_from_neighbor
                        neighbor_node.vel = initial_velocity
                        neighbor_node.successor = current_id
                        neighbor_node.path_geometry = path_geometry  # Store the path geometry
                        neighbor_node.calc_circle_centers(rho=self.rho, vmax=self.v_max)
                        
                        # Update its priority in the queue with the new, lower cost
                        counter += 1
                        heapq.heappush(pq, (new_cost_from_neighbor, counter, neighbor_id))
        
        return node_dict

    

    def _find_connecting_segment(self, start_pos, target_node):
        """
        Finds the shortest, obstacle-free, kinodynamically feasible path
        from a start position to a target node's state (pos, vel, circles).
        This version is more permissive with tangent validation.
        """
        if target_node.circle_cw is None or target_node.circle_ccw is None:
            return None
            
        possible_paths = []
        
        for circle_type, circle_center in [('cw', target_node.circle_cw), ('ccw', target_node.circle_ccw)]:
            tangents = find_tangents_to_circle(start_pos, circle_center, self.rho)
            
            for tangent_point in tangents:
                # Skip if tangent point is too close to start point
                if np.linalg.norm(np.array(tangent_point) - np.array(start_pos)) < 1e-9:
                    continue
                    
                # Check for obstruction with more permissive parameters
                if self._is_kinodynamic_path_obstructed(
                    start_pos, tangent_point, target_node.pos, 
                    circle_center, circle_type,
                    segments=8  # Fewer segments for faster checking
                ):
                    continue

                # Calculate path lengths
                straight_len = np.linalg.norm(np.array(tangent_point) - np.array(start_pos))
                arc_len = calculate_arc_length(
                    tangent_point, target_node.pos, 
                    circle_center, self.rho, circle_type
                )
                total_len = straight_len + arc_len
                
                # Calculate initial velocity (normalized direction)
                direction = np.array(tangent_point) - np.array(start_pos)
                if np.linalg.norm(direction) > 1e-9:
                    initial_vel = tuple((direction / np.linalg.norm(direction)) * self.v_max)
                    
                    # Create path geometry information
                    path_geom = {'type': circle_type, 'tangent_point': tangent_point}
                    possible_paths.append((total_len, initial_vel, path_geom))

        if not possible_paths:
            return None
        
        # Return the path with the minimum total length
        return min(possible_paths, key=lambda x: x[0])


    # Place this inside your MILPTrajectoryPlanner class

    # Place this inside your MILPTrajectoryPlanner class

    def _is_kinodynamic_path_obstructed(self, start_pos, tangent_point, target_pos, circle_center, circle_type, segments=10):
        """
        A robust check to see if the entire kinodynamic path is obstructed,
        with a special case to allow travel along obstacle edges between adjacent vertices.
        (Corrected version to fix AttributeError)
        """
        # 1. Check the straight line segment
        line = LineString([start_pos, tangent_point])
        if line.length > 1e-9:
            is_an_edge = False
            # Check if the line segment is an edge of any obstacle
            for obs_verts in self.obstacles:
                for i in range(len(obs_verts)):
                    p1 = obs_verts[i]
                    p2 = obs_verts[(i + 1) % len(obs_verts)]
                    edge = LineString([p1, p2])

                    # --- KEY FIX: Use .coords to access endpoints ---
                    # This block replaces the incorrect .start and .end attributes.
                    # It checks if the `line` and `edge` have the same endpoints,
                    # regardless of their direction.
                    line_start = Point(line.coords[0])
                    line_end = Point(line.coords[-1])
                    edge_start = Point(edge.coords[0])
                    edge_end = Point(edge.coords[-1])

                    if (line_start.distance(edge_start) < 1e-9 and line_end.distance(edge_end) < 1e-9) or \
                    (line_start.distance(edge_end) < 1e-9 and line_end.distance(edge_start) < 1e-9):
                        is_an_edge = True
                        break
                if is_an_edge:
                    break
            
            # If the line is not an obstacle edge, perform the standard interior check
            if not is_an_edge:
                for polygon in self.obstacle_polygons:
                    if polygon.buffer(-1e-9).intersects(line):
                        return True

        # 2. Check the arc segment (this logic remains the same)
        arc_points = approximate_arc(
            tangent_point, target_pos, circle_center, 
            self.rho, circle_type, segments
        )
        
        for i in range(len(arc_points) - 1):
            segment_line = LineString([arc_points[i], arc_points[i+1]])
            if segment_line.length < 1e-9: continue
                
            for polygon in self.obstacle_polygons:
                if polygon.buffer(-1e-9).intersects(segment_line):
                    return True
                    
        return False


    def _get_valid_obstacle_vertices(self):
        """Helper to extract non-obstructed obstacle vertices."""
        valid_vertices = []
        all_vertices_set = set()
        for obstacle in self.obstacles:
            for vertex in obstacle:
                all_vertices_set.add(tuple(vertex))

        for vertex_tuple in all_vertices_set:
            point = Point(vertex_tuple)
            is_inside_other = False
            for poly in self.obstacle_polygons:
                # Check if the vertex is strictly inside another polygon
                # A small buffer can help with floating point issues
                if poly.buffer(-1e-9).contains(point):
                    is_inside_other = True
                    break
            if not is_inside_other:
                valid_vertices.append(vertex_tuple)
        return valid_vertices