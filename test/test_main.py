""" Tests for the main module. """
import sys
import pytest

sys.path.append('.')
from main import TrajectoryDesign, is_path_blocked, lines_intersect


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

# @pytest.fixture(scope='module')
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





def test_intersecting_lines():
    # Lines intersect
    assert lines_intersect([0, 0], [2, 2], [0, 2], [2, 0]) is True

def test_non_intersecting_parallel_lines():
    # Parallel lines do not intersect
    assert lines_intersect([0, 0], [2, 0], [0, 1], [2, 1]) is False

def test_non_intersecting_non_parallel_lines():
    # Non-parallel lines that do not cross
    assert lines_intersect([0, 0], [1, 1], [2, 2], [3, 3]) is False



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

    
