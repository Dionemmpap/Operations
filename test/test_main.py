import sys
sys.path.append('.')


from main import TrajectoryDesign, is_path_blocked
import pytest

#helper function to calculate distance between two points
def distance(point1, point2):
    
    return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5

#Variables
map_boundary = [[0, 0], [10, 0], [10, 10], [0, 10]]
obstacles = [[[2, 2], [4, 2], [4, 4], [2, 4]], [[6, 6], [8, 6], [8, 8], [6, 8]]]
end_point = [9, 9]
start_point = [0, 0]
tau = 1

# @pytest.fixture(scope='module')
def test_build_graph():
    td = TrajectoryDesign(map_boundary, obstacles, end_point, start_point, tau)
    graph, points = td.build_graph()

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



    #Check that the computed distances are correct