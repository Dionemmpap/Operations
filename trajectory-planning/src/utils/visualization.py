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