import matplotlib.pyplot as plt


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


def visualize_nodes(node_dict, ax=None, rho=1.0):
    """
    Visualize the nodes in the trajectory planning tree.
    
    Args:
        node_dict: Dictionary of nodes from build_feasible_traj_tree
        ax: Matplotlib axis to plot on (if None, creates a new figure)
        rho: The turning radius for circle drawing
    """
    # Create new figure if none provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 9))
        standalone_figure = True
    else:
        standalone_figure = False
    
    # Count reachable nodes for reporting
    reachable = 0
    unreachable = 0
    
    # Plot each node in the dictionary
    for node_id, node_obj in node_dict.items():
        # Different styling based on node type
        if node_id == 'start':
            ax.scatter(node_obj.pos[0], node_obj.pos[1], color='green', s=100, zorder=10, label='Start')
        elif node_id == 'goal':
            ax.scatter(node_obj.pos[0], node_obj.pos[1], color='red', s=100, zorder=10, label='Goal')
        elif node_obj.cost_to_go < float('inf'):
            ax.scatter(node_obj.pos[0], node_obj.pos[1], color='blue', s=50, zorder=5)
            reachable += 1
            
            # Draw an arrow for the velocity direction if available
            if node_obj.vel:
                vel_scale = 2.0  # Scale factor for better visualization
                dx, dy = node_obj.vel[0] * vel_scale, node_obj.vel[1] * vel_scale
                ax.arrow(node_obj.pos[0], node_obj.pos[1], dx, dy, 
                         head_width=0.3, head_length=0.5, fc='green', ec='green', alpha=0.7)
        else:
            # Unreachable nodes
            ax.scatter(node_obj.pos[0], node_obj.pos[1], color='gray', s=30, alpha=0.5)
            unreachable += 1
        
        # Show node ID for reference
        ax.annotate(node_id, (node_obj.pos[0], node_obj.pos[1]), 
                   xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Draw the turning circles
        if node_obj.circle_cw is not None:
            circle = plt.Circle(node_obj.circle_cw, radius=rho, 
                              color='blue', fill=False, alpha=0.3, linestyle='--')
            ax.add_patch(circle)
            
        if node_obj.circle_ccw is not None:
            circle = plt.Circle(node_obj.circle_ccw, radius=rho, 
                              color='red', fill=False, alpha=0.3, linestyle='--')
            ax.add_patch(circle)
    
    # Add title with statistics
    if standalone_figure:
        ax.set_title(f"Trajectory Tree Nodes (Reachable: {reachable}, Unreachable: {unreachable})")
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Create custom legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Start'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Goal'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Reachable Node'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=6, label='Unreachable Node'),
            Line2D([0], [0], color='blue', linestyle='--', label='CW Circle'),
            Line2D([0], [0], color='red', linestyle='--', label='CCW Circle'),
            Line2D([0], [0], color='green', label='Velocity')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.show()
    
    return reachable, unreachable