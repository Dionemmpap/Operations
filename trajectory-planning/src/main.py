""" Receding Horizon Control for Trajectory Design """
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from planners.milp_planner2 import MILPTrajectoryPlanner
from utils.visualization import visualize_map
from utils.geometry import approximate_arc

def test_trajectory_tree():
    """Test the trajectory tree generation with corrected visualization."""
    # --- 1. Load Map and Initialize Planner (Your code is correct here) ---
    map_path = Path(__file__).parent.parent / 'maps' / 'scenarios' / 'paper_validation.json'
    with open(map_path, 'r') as f:
        map_data = json.load(f)
    
    map_boundary = map_data['map_boundary']
    obstacles = map_data['obstacles']
    start_point = map_data['start_point']
    end_point = map_data['end_point']
    
    goal_state = {'pos': end_point, 'vel': (1.0, 0.0)} # Arrive heading left
    
    planner = MILPTrajectoryPlanner(
        map_boundary=map_boundary,
        obstacles=obstacles,
        goal_state=[end_point],
        ini_state=[start_point],
        v_max=1.0, turning_radius=1.0, Delta_T=0.1, N=10, N_exec=5
    )
    
    # --- 2. Build the Tree (Your code is correct here) ---
    print("Building trajectory tree...")
    node_dict = planner.build_feasible_traj_tree(
        start_pos=start_point,
        goal_state=goal_state
    )
    print(f"Tree generation complete. Found {len(node_dict)} nodes.")
    print_node_dict(node_dict)

    # --- 3. Visualize the FULL Kinodynamic Tree (Corrected Logic) ---
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Plot obstacles and boundary
    for obstacle in obstacles:
        ax.add_patch(plt.Polygon(obstacle, closed=True, color='darkgray'))
    ax.plot(*zip(*(map_boundary + [map_boundary[0]])), 'k--') # Plot boundary

    # Plot the full tree edges (the kinodynamic paths)
    for node_obj in node_dict.values():
        # Check if the node has a valid path to a successor
        if node_obj.successor and hasattr(node_obj, 'path_geometry'):
            succ_node = node_dict[node_obj.successor]
            path_geom = node_obj.path_geometry
            tangent_point = path_geom['tangent_point']
            circle_type = path_geom['type']

            # --- KEY FIX 1: Use the SUCCESSOR's circle center ---
            # The path connects TO the successor's turning circle.
            if circle_type == 'cw':
                circle_center = succ_node.circle_cw
            else:
                circle_center = succ_node.circle_ccw

            if circle_center is None: continue

            # a) Draw the straight line segment
            ax.plot([node_obj.pos[0], tangent_point[0]], 
                    [node_obj.pos[1], tangent_point[1]], 
                    'b-', alpha=0.4, linewidth=0.8)
            
            # b) Draw the circular arc segment
            arc_points = approximate_arc(tangent_point, succ_node.pos, 
                                        circle_center, planner.rho, circle_type, 15)
            ax.plot(*zip(*arc_points), 'r-', alpha=0.4, linewidth=0.8)

    # Plot nodes on top of the paths
    for node_id, node_obj in node_dict.items():
        if node_id == 'start':
            ax.plot(node_obj.pos[0], node_obj.pos[1], 'go', markersize=10, zorder=5)
        elif node_id == 'goal':
            ax.plot(node_obj.pos[0], node_obj.pos[1], 'ro', markersize=10, zorder=5)
        elif node_obj.cost_to_go < float('inf'):
            ax.plot(node_obj.pos[0], node_obj.pos[1], 'bo', markersize=4, zorder=5)

    ax.set_title('Full Kinodynamic Trajectory Tree')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True)
    ax.axis('equal')
    plt.show()

    # --- 4. Visualize the SINGLE OPTIMAL PATH (Corrected Logic) ---
    if node_dict['start'].cost_to_go < float('inf'):
        print(f"Optimal path from start found! Total Cost: {node_dict['start'].cost_to_go:.2f}")
        fig, ax = plt.subplots(figsize=(16, 12))

        # Plot obstacles and boundary
        for obstacle in obstacles:
            ax.add_patch(plt.Polygon(obstacle, closed=True, color='darkgray'))
        ax.plot(*zip(*(map_boundary + [map_boundary[0]])), 'k--')

        # Trace the path from start to goal
        current_id = 'start'
        while current_id != 'goal' and node_dict[current_id].successor:
            node_obj = node_dict[current_id]
            succ_node = node_dict[node_obj.successor]
            
            # Draw the kinodynamic segment for this step in the path
            path_geom = node_obj.path_geometry
            tangent_point = path_geom['tangent_point']
            circle_type = path_geom['type']

            # --- KEY FIX 2: Draw the straight+arc path, not just a line ---
            if circle_type == 'cw':
                circle_center = succ_node.circle_cw
            else:
                circle_center = succ_node.circle_ccw
            
            # Draw circles for context
            if succ_node.circle_cw: ax.add_patch(plt.Circle(succ_node.circle_cw, planner.rho, fill=False, color='blue', ls='--', alpha=0.5))
            if succ_node.circle_ccw: ax.add_patch(plt.Circle(succ_node.circle_ccw, planner.rho, fill=False, color='red', ls='--', alpha=0.5))
            
            # Draw straight segment
            ax.plot([node_obj.pos[0], tangent_point[0]], 
                    [node_obj.pos[1], tangent_point[1]], 
                    'b-', lw=2)
            
            # Draw arc segment
            arc_points = approximate_arc(tangent_point, succ_node.pos, 
                                        circle_center, planner.rho, circle_type, 20)
            ax.plot(*zip(*arc_points), 'r-', lw=2)

            # Move to the next node in the path
            current_id = node_obj.successor

        # Plot nodes on the path
        ax.plot(start_point[0], start_point[1], 'go', markersize=12, label='Start')
        ax.plot(end_point[0], end_point[1], 'ro', markersize=12, label='Goal')

        ax.set_title('Optimal Path from Start to Goal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True)
        ax.axis('equal')
        ax.legend()
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

def print_node_dict(node_dict):
    """Print the node dictionary in a readable format."""
    print("\n=== Node Dictionary ===")
    print(f"Total nodes: {len(node_dict)}")
    
    # Group nodes by type
    start_node = node_dict.get('start')
    goal_node = node_dict.get('goal')
    obstacle_nodes = {k: v for k, v in node_dict.items() if k not in ['start', 'goal']}
    
    # Print start and goal nodes
    if start_node:
        cost_str = f"{start_node.cost_to_go:.2f}" if start_node.cost_to_go != float('inf') else "inf"
        print(f"Start node: pos={start_node.pos}, cost={cost_str}, successor={start_node.successor}")
    
    if goal_node:
        cost_str = f"{goal_node.cost_to_go:.2f}" if goal_node.cost_to_go != float('inf') else "inf"
        print(f"Goal node: pos={goal_node.pos}, cost={cost_str}, successor={goal_node.successor}")
    
    # Print obstacle nodes
    reachable_obs = 0
    unreachable_obs = 0
    
    print("\nObstacle Nodes:")
    print("ID\tPosition\t\tCost\t\tSuccessor")
    print("-" * 50)
    
    for node_id, node_obj in sorted(obstacle_nodes.items()):
        if node_obj.cost_to_go < float('inf'):
            status = "REACHABLE"
            reachable_obs += 1
            cost_str = f"{node_obj.cost_to_go:.2f}"
        else:
            status = "UNREACHABLE"
            unreachable_obs += 1
            cost_str = "inf"
            
        print(f"{node_id}\t{node_obj.pos}\t{cost_str}\t{node_obj.successor}\t{status}")
    
    print(f"\nReachable obstacle nodes: {reachable_obs}/{len(obstacle_nodes)}")
    print(f"Unreachable obstacle nodes: {unreachable_obs}/{len(obstacle_nodes)}")
    print("=======================")

if __name__ == "__main__":
    # Run the test for a single connection
    # test_connecting_segment()
    
    # Run the test for trajectory tree generation
    test_trajectory_tree()

