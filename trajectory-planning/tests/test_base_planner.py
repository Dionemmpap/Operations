""" Tests for the main module. """
import sys
import pytest
import numpy as np
from shapely.geometry import Polygon

sys.path.append('.')
from src.utils.geometry import is_path_blocked
from src.utils.obstacles import get_obstacles, merge_intersecting_obstacles
from src.planners.base_planner import TrajectoryDesignBase as TrajectoryDesign


#helper function to calculate distance between two points
def distance(point1, point2):
    """ Calculate the Euclidean distance between two points. """
    return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5

#Variables
map_boundary = [[0, 0], [10, 0], [10, 10], [0, 10]]
obstacles = [[[2, 2], [4, 2], [4, 4], [2, 4]], [[6, 6], [8, 6], [8, 8], [6, 8]]]
end_point = [9, 9]
start_point = [0, 0]
tau = 1


def test_build_graph():
    """ Test the build_graph method by checking the distances between the generated nodes. """	
    td = TrajectoryDesign(map_boundary, obstacles, end_point, start_point, tau)
    graph, points = td.build_graph()
    assert len(points) == 9
    assert len(graph) == 9
    for point1 in points:
        for point2 in points:
            if point1 != point2 and not is_path_blocked(point1, point2, obstacles):
                assert graph[tuple(point1)][tuple(point2)] == distance(point1, point2)



def test_dijkstra():
    td = TrajectoryDesign(map_boundary, obstacles, end_point, start_point, tau)
    graph, points = td.build_graph()
    distances = td.dijkstra()
    assert distances[(8, 8)] == 2**0.5
    assert distances[(2,2)] == 2+ 32**0.5 + 10**0.5
    assert distances[(4,2)] == 32**0.5 + 10**0.5
    assert distances[(4,4)] == 20**0.5 + 10**0.5
    assert distances[(2,4)] == 32**0.5 + 10**0.5
    assert distances[(6,6)] == 2 + 10**0.5
    assert distances[(8,6)] == 10**0.5
    assert distances[(6,8)] == 10**0.5



def test_get_obstacles():
    np.random.seed(42)  # Seed to ensure deterministic results
    map_boundary = [(0, 0), (10, 0), (0, 10)]  # Example map boundary
    test_obstacle = get_obstacles(map_boundary, 1)
    
    # Bottom-left corner coordinates
    x1, y1 = test_obstacle[0][0]

    # Width and height
    width = test_obstacle[0][1][0] - x1
    height = test_obstacle[0][2][1] - y1

    # Assert bottom-left corner is within 15% and 75% of map boundary dimensions
    assert 0.15 * map_boundary[1][0] <= x1 <= 0.75 * map_boundary[1][0]
    assert 0.15 * map_boundary[2][1] <= y1 <= 0.75 * map_boundary[2][1]

    # Assert width and height are within 10% and 25% of map boundary dimensions
    assert 0.1 * map_boundary[1][0] <= width <= 0.25 * map_boundary[1][0]
    assert 0.1 * map_boundary[2][1] <= height <= 0.25 * map_boundary[2][1]



def test_merge_intersecting_obstacles():
    obstacles = [
        [[1, 1], [3, 1], [3, 3], [1, 3]],  # Square 1
        [[2, 2], [4, 2], [4, 4], [2, 4]]   # Square 2 (intersects with Square 1)
    ]
    
    merged_obstacles = merge_intersecting_obstacles(obstacles)
    
    # Assert only one merged obstacle is returned
    assert len(merged_obstacles) == 1

    # Assert merged obstacle contains the expected bounding box
    merged_polygon = Polygon(merged_obstacles[0])
    expected_polygon = Polygon([[1, 1], [3, 1], [3, 2], [4, 2], [4, 4], [2, 4], [2, 3], [1, 3]])
    assert merged_polygon.equals(expected_polygon)


def test_path_blocked_by_single_obstacle():
    # Path intersects an obstacle
    obstacles = [[[1, 1], [3, 1], [3, 3], [1, 3]]]  # Square obstacle
    assert is_path_blocked([0, 0], [4, 4], obstacles) is True

def test_path_not_blocked():
    # Path does not intersect any obstacles
    obstacles = [[[1, 1], [3, 1], [3, 3], [1, 3]]]  # Square obstacle
    assert is_path_blocked([0, 0], [0, 4], obstacles) is False

def test_path_blocked_by_multiple_obstacles():
    # Path intersects one of multiple obstacles
    obstacles = [
        [[1, 1], [3, 1], [3, 3], [1, 3]],  # Obstacle 1
        [[5, 5], [7, 5], [7, 7], [5, 7]],  # Obstacle 2
    ]
    assert is_path_blocked([0, 0], [6, 6], obstacles) is True

def test_path_with_no_obstacles():
    # No obstacles; path is not blocked
    obstacles = []
    assert is_path_blocked([0, 0], [4, 4], obstacles) is False

def test_path_tangent_to_obstacle():
    # Path is tangent to the obstacle
    obstacles = [[[1, 1], [3, 1], [3, 3], [1, 3]]]  # Square obstacle
    assert is_path_blocked([0, 0], [4, 1], obstacles) is False


def test_plan_trajectory():
    """Test the trajectory planning logic."""
    # Set up the test data
    current_position = [0, 0]
    end_point = [10, 10]
    obstacles = [[[2, 2], [4, 2], [4, 4], [2, 4]]]
    tau = 0.1

    td = TrajectoryDesign(map_boundary, obstacles, end_point, start_point, tau)
    point = td.plan_trajectory(current_position)

    # Assertions
    assert isinstance(point, np.ndarray)
    assert len(point) == 2

    # Ensure the proposed point moves closer to the end point
    initial_distance = np.linalg.norm(np.array(current_position) - np.array(end_point))
    new_distance = np.linalg.norm(np.array(point) - np.array(end_point))
    assert new_distance < initial_distance
    
def test_receding_horizon():
    """Test the receding horizon trajectory planning logic."""
    # Set up the test data
    end_point = [10, 10]
    obstacles = [[[2, 2], [4, 2], [4, 4], [2, 4]]]
    tau = 0.1

    td = TrajectoryDesign(map_boundary, obstacles, end_point, start_point, tau)
    td.receding_horizon()


    # Assertions
    assert isinstance(td.trajectory, list)
    assert all(isinstance(point, np.ndarray) for point in td.trajectory)
    assert len(td.trajectory) > 0
    assert all(len(point) == 2 for point in td.trajectory)
    assert all(isinstance(point[0], (int, float, np.int32)) and isinstance(point[1], (int, float, np.int32)) for point in td.trajectory)
    assert all(0 <= point[0] <= 10 and 0 <= point[1] <= 10 for point in td.trajectory)
    diff_traj = np.diff(td.trajectory, axis=0)
    assert np.all(np.diff(td.trajectory, axis=0) > 0)  # Ensure the trajectory is monotonically increasing