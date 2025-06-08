""" Receding Horizon Control for Trajectory Design """
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from planners.milp_planner2 import MILPTrajectoryPlanner
from utils.visualization import visualize_map
from utils.geometry import approximate_arc

def test_trajectory_tree():
    """Test the trajectory tree generation."""
    # Load map data
    map_path = Path(__file__).parent.parent / 'maps' / 'scenarios' / 'basic_map.json'
    
    if not map_path.exists():
        print(f"Error: Map file not found at {map_path}")
        # Try basic_map as fallback
        map_path = Path(__file__).parent.parent / 'maps' / 'scenarios' / 'basic_map.json'
        if not map_path.exists():
            print(f"Error: Fallback map file not found at {map_path}")
            return
    
    with open(map_path, 'r') as f:
        map_data = json.load(f)
    
    # Extract map data
    map_boundary = map_data['map_boundary']
    obstacles = map_data['obstacles']
    start_point = map_data['start_point']
    end_point = map_data['end_point']
    
    print(f"Loaded map: {map_data['name']}")
    print(f"Obstacles: {len(obstacles)}")
    
    # Visualize the environment
    visualize_map(map_boundary, obstacles, {}, end_point)
    
    # Create goal and initial states
    # For Bellingham's approach, we need velocity information
    goal_state = {'pos': end_point, 'vel': (1.0, 0.0)}  # Assume moving right at goal
    ini_state = {'pos': start_point, 'vel': (1.0, 0.0)}  # Assume moving right at start
    
    # Initialize the planner
    v_max = 1.0  # Maximum velocity
    turning_radius = 2.0  # Minimum turning radius
    Delta_T = 0.1  # Time step
    N = 10  # Horizon length
    N_exec = 5  # Number of steps to execute
    
    planner = MILPTrajectoryPlanner(
        map_boundary=map_boundary,
        obstacles=obstacles,
        goal_state=[end_point],  # Format expected by your constructor
        ini_state=[start_point],  # Format expected by your constructor
        v_max=v_max,
        turning_radius=turning_radius,
        Delta_T=Delta_T,
        N=N,
        N_exec=N_exec
    )
    
    # Build the trajectory tree
    print("Building trajectory tree...")
    node_dict = planner.build_feasible_traj_tree(
        start_pos=start_point,
        goal_state=goal_state
    )
    
    # Print results
    print(f"Tree generation complete. Found {len(node_dict)} nodes.")
    reachable_nodes = sum(1 for n in node_dict.values() if n.cost_to_go < float('inf'))
    print(f"Reachable nodes: {reachable_nodes}")
    
    # ===== Visualization of Tree =====
    plt.figure(figsize=(15, 10))
    
    # Plot obstacles
    for obstacle in obstacles:
        obstacle_x = [p[0] for p in obstacle] + [obstacle[0][0]]  # Close the polygon
        obstacle_y = [p[1] for p in obstacle] + [obstacle[0][1]]
        plt.plot(obstacle_x, obstacle_y, 'k-', linewidth=1)
    
    # Plot turning circles
    circle_count = 0
    circle_limit = 30  # Limit number of circles to avoid clutter
    
    # First plot circles for special nodes (start and goal)
    for special_id in ['start', 'goal']:
        if special_id in node_dict:
            node_obj = node_dict[special_id]
            
            # Only draw circles for nodes with valid velocities
            if node_obj.vel is not None and node_obj.cost_to_go < float('inf'):
                # Draw CW circle
                if node_obj.circle_cw is not None:
                    circle = plt.Circle(node_obj.circle_cw, planner.rho, 
                                      fill=False, color='blue', alpha=0.5, linestyle='--')
                    plt.gca().add_patch(circle)
                    plt.plot(node_obj.circle_cw[0], node_obj.circle_cw[1], 'bx', markersize=4)
                
                # Draw CCW circle
                if node_obj.circle_ccw is not None:
                    circle = plt.Circle(node_obj.circle_ccw, planner.rho, 
                                      fill=False, color='red', alpha=0.5, linestyle='--')
                    plt.gca().add_patch(circle)
                    plt.plot(node_obj.circle_ccw[0], node_obj.circle_ccw[1], 'rx', markersize=4)
                
                # Show velocity vector
                arrow_len = 1.0  # Scale factor for better visibility
                plt.arrow(node_obj.pos[0], node_obj.pos[1], 
                          node_obj.vel[0]*arrow_len, node_obj.vel[1]*arrow_len,
                          head_width=0.2, head_length=0.3, fc='green', ec='green')
    
    # Then draw for a limited number of other nodes with valid costs
    for node_id, node_obj in node_dict.items():
        if node_id not in ['start', 'goal'] and node_obj.cost_to_go < float('inf'):
            if circle_count >= circle_limit:
                break
                
            if node_obj.circle_cw is not None:
                circle = plt.Circle(node_obj.circle_cw, planner.rho, 
                                  fill=False, color='blue', alpha=0.2, linestyle=':')
                plt.gca().add_patch(circle)
                
            if node_obj.circle_ccw is not None:
                circle = plt.Circle(node_obj.circle_ccw, planner.rho, 
                                  fill=False, color='red', alpha=0.2, linestyle=':')
                plt.gca().add_patch(circle)
                
            circle_count += 1
    
    # Plot tree edges (paths between nodes)
    for node_id, node_obj in node_dict.items():
        if hasattr(node_obj, 'successor') and node_obj.successor and hasattr(node_obj, 'path_geometry') and node_obj.path_geometry:
            succ_id = node_obj.successor
            if succ_id in node_dict:
                succ_node = node_dict[succ_id]
                
                # Extract path geometry
                path_geom = node_obj.path_geometry
                tangent_point = path_geom['tangent_point']
                circle_type = path_geom['type']
                
                # 1. Draw the straight line from node to tangent point
                plt.plot([node_obj.pos[0], tangent_point[0]], 
                         [node_obj.pos[1], tangent_point[1]], 
                         'b-', alpha=0.5, linewidth=0.5)
                
                # 2. Draw the circular arc from tangent point to successor
                # Determine which circle to use
                if circle_type == 'cw':
                    circle_center = node_obj.circle_cw
                else:  # 'ccw'
                    circle_center = node_obj.circle_ccw
                    
                # Approximate the arc with points
                arc_points = approximate_arc(tangent_point, succ_node.pos, 
                                            circle_center, planner.rho, circle_type, 15)
                
                # Draw the arc as a series of line segments
                for i in range(len(arc_points)-1):
                    plt.plot([arc_points[i][0], arc_points[i+1][0]], 
                             [arc_points[i][1], arc_points[i+1][1]], 
                             'b-', alpha=0.5, linewidth=0.5)

    
    # Plot nodes
    for node_id, node_obj in node_dict.items():
        if node_id == 'start':
            plt.plot(node_obj.pos[0], node_obj.pos[1], 'go', markersize=10, label='Start')
        elif node_id == 'goal':
            plt.plot(node_obj.pos[0], node_obj.pos[1], 'ro', markersize=10, label='Goal')
        elif node_obj.cost_to_go < float('inf'):  # Only plot reachable nodes
            plt.plot(node_obj.pos[0], node_obj.pos[1], 'bo', markersize=3)
    
    # Plot map boundary
    boundary_x = [p[0] for p in map_boundary] + [map_boundary[0][0]]
    boundary_y = [p[1] for p in map_boundary] + [map_boundary[0][1]]
    plt.plot(boundary_x, boundary_y, 'k--', linewidth=1)
    
    # Create legend with custom elements
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='g', markersize=10, label='Start'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=10, label='Goal'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='b', markersize=6, label='Vertex Node'),
        Line2D([0], [0], color='blue', lw=1, linestyle='--', label='CW Turning Circle'),
        Line2D([0], [0], color='red', lw=1, linestyle='--', label='CCW Turning Circle'),
        Line2D([0], [0], color='green', lw=1, label='Velocity Vector')
    ]
    plt.legend(handles=legend_elements, loc='upper left')
    
    plt.title('Trajectory Tree with Turning Circles')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
    
    # ===== Plot optimal path if found =====
    if 'start' in node_dict and node_dict['start'].cost_to_go < float('inf'):
        print(f"Path found! Cost: {node_dict['start'].cost_to_go:.2f}")
        
        # Trace path from start to goal
        current_id = 'start'
        path = [node_dict[current_id].pos]
        path_nodes = [current_id]
        
        while hasattr(node_dict[current_id], 'successor') and node_dict[current_id].successor:
            current_id = node_dict[current_id].successor
            path.append(node_dict[current_id].pos)
            path_nodes.append(current_id)
            if current_id == 'goal':
                break
        
        # Plot the path
        plt.figure(figsize=(15, 10))
        
        # Plot obstacles
        for obstacle in obstacles:
            obstacle_x = [p[0] for p in obstacle] + [obstacle[0][0]]
            obstacle_y = [p[1] for p in obstacle] + [obstacle[0][1]]
            plt.plot(obstacle_x, obstacle_y, 'k-', linewidth=1)
        
        # Plot turning circles for nodes in the path
        for i in range(len(path_nodes)-1):
            node_id = path_nodes[i]
            node_obj = node_dict[node_id]
            next_node = node_dict[path_nodes[i+1]]
            
            # Draw circles
            if node_obj.circle_cw is not None:
                circle = plt.Circle(node_obj.circle_cw, planner.rho, 
                                  fill=False, color='blue', alpha=0.3, linestyle='--')
                plt.gca().add_patch(circle)
                
            if node_obj.circle_ccw is not None:
                circle = plt.Circle(node_obj.circle_ccw, planner.rho, 
                                  fill=False, color='red', alpha=0.3, linestyle='--')
                plt.gca().add_patch(circle)
            
            # Draw velocity vector
            if node_obj.vel is not None:
                arrow_len = 1.0
                plt.arrow(node_obj.pos[0], node_obj.pos[1], 
                          node_obj.vel[0]*arrow_len, node_obj.vel[1]*arrow_len,
                          head_width=0.2, head_length=0.3, fc='green', ec='green')
        
        # Plot path
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        plt.plot(path_x, path_y, 'g-', linewidth=2, label='Optimal Path')
        
        # Mark nodes on path
        for i, node_id in enumerate(path_nodes):
            pos = node_dict[node_id].pos
            if i == 0:
                plt.plot(pos[0], pos[1], 'go', markersize=10, label='Start')
            elif i == len(path_nodes)-1:
                plt.plot(pos[0], pos[1], 'ro', markersize=10, label='Goal')
            else:
                plt.plot(pos[0], pos[1], 'bo', markersize=6)
                
        # Plot map boundary
        plt.plot(boundary_x, boundary_y, 'k--', linewidth=1)
        
        plt.title('Optimal Trajectory Path')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        plt.legend()
        plt.axis('equal')
        plt.tight_layout()
        plt.show()
    else:
        print("No path found from start to goal.")
def test_connecting_segment():
    """Test the connection between two trajectory nodes."""
    # Inside a test script
    from planners.milp_planner2 import MILPTrajectoryPlanner
    from utils.obstacles import node
    import matplotlib.pyplot as plt
    import numpy as np
    from utils.geometry import approximate_arc
    
    # Create a simple test environment
    planner = MILPTrajectoryPlanner(
        map_boundary=[(0, 0), (10, 0), (10, 10), (0, 10)],
        obstacles=[[(2, 2), (3, 2), (3, 3), (2, 3)]],
        goal_state=[(8, 8)],  # Goal position
        ini_state=[(1, 1)],   # Initial position
        v_max=1.0,
        turning_radius=1.0,
        Delta_T=0.1,
        N=10,
        N_exec=5
    )
    
    # Create and set up the goal node
    goal_node = node(pos=(8, 8), vel=(1.0, 0.0), cost_to_go=0)
    goal_node.calc_circle_centers(rho=planner.rho, vmax=planner.v_max)
    
    # Try multiple start positions to test different configurations
    for test_pos in [(3, 5), (5, 3), (8, 2)]:
        print("\n==== Testing start position: ", test_pos, " ====")
        # Set the start position
        start_pos = test_pos
        
        # Find a connecting segment from start to goal
        connection = planner._find_connecting_segment(start_pos, goal_node)
        print(f"Connection found: {connection}")
        
        if connection:
            path_length, initial_velocity, path_geometry = connection
            print(f"Path length: {path_length}")
            print(f"Initial velocity: {initial_velocity}")
            print(f"Path geometry: {path_geometry}")
            
            # Visualize the connection
            plt.figure(figsize=(10, 8))
            
            # Draw obstacles
            for obstacle in planner.obstacles:
                obstacle_x = [p[0] for p in obstacle] + [obstacle[0][0]]
                obstacle_y = [p[1] for p in obstacle] + [obstacle[0][1]]
                plt.plot(obstacle_x, obstacle_y, 'k-', linewidth=1)
            
            # Draw the turning circles
            if goal_node.circle_cw is not None:
                circle = plt.Circle(goal_node.circle_cw, planner.rho, 
                                  fill=False, color='blue', alpha=0.5, linestyle='--')
                plt.gca().add_patch(circle)
                plt.plot(goal_node.circle_cw[0], goal_node.circle_cw[1], 'bx', markersize=4)
            
            if goal_node.circle_ccw is not None:
                circle = plt.Circle(goal_node.circle_ccw, planner.rho, 
                                  fill=False, color='red', alpha=0.5, linestyle='--')
                plt.gca().add_patch(circle)
                plt.plot(goal_node.circle_ccw[0], goal_node.circle_ccw[1], 'rx', markersize=4)
            
            # Draw the start and goal points
            plt.plot(start_pos[0], start_pos[1], 'go', markersize=10, label='Start')
            plt.plot(goal_node.pos[0], goal_node.pos[1], 'ro', markersize=10, label='Goal')
            
            # Draw the path
            tangent_point = path_geometry['tangent_point']
            circle_type = path_geometry['type']
            
            # Draw the straight line segment
            plt.plot([start_pos[0], tangent_point[0]], 
                     [start_pos[1], tangent_point[1]], 
                     'b-', linewidth=2, label='Straight segment')
            
            # Draw the circular arc segment
            circle_center = goal_node.circle_cw if circle_type == 'cw' else goal_node.circle_ccw
            arc_points = approximate_arc(tangent_point, goal_node.pos, 
                                        circle_center, planner.rho, circle_type, 20)
            
            arc_x = [p[0] for p in arc_points]
            arc_y = [p[1] for p in arc_points]
            plt.plot(arc_x, arc_y, 'r-', linewidth=2, label='Arc segment')
            
            # Draw initial velocity vector
            plt.arrow(start_pos[0], start_pos[1], 
                      initial_velocity[0], initial_velocity[1],
                      head_width=0.2, head_length=0.3, fc='green', ec='green',
                      label='Initial velocity')
            
            plt.title('Kinodynamic Connection Test')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.grid(True)
            plt.axis('equal')
            plt.legend()
            plt.tight_layout()
            plt.show()
        else:
            print("No connection found!")

if __name__ == "__main__":
    # Run the test for a single connection
    test_connecting_segment()
    
    # Run the test for trajectory tree generation
    # test_trajectory_tree()

