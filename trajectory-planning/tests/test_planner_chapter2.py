import pytest
import numpy as np
import sys
from pathlib import Path

from planners.planner_chapter2 import RecedingHorizonController

@pytest.fixture
def simple_environment():
    """Create a simple test environment with known parameters."""
    map_boundary = [[0, 0], [0, 10], [10, 10], [10, 0]]
    obstacles = [[[2, 2], [2, 4], [4, 4], [4, 2]]]  # One square obstacle
    start_point = [1, 1]
    end_point = [9, 9]
    return map_boundary, obstacles, start_point, end_point

@pytest.fixture
def controller(simple_environment):
    """Create a controller instance with simplified parameters for testing."""
    map_boundary, obstacles, start_point, end_point = simple_environment
    # Disable visualizer for testing and use smaller horizon for speed
    return RecedingHorizonController(
        map_boundary, obstacles, start_point, end_point, 
        N=10, Ne=2, tau=0.5, umax=1.0, use_visualizer=False
    )

def test_initialization(simple_environment):
    """Test that the controller initializes correctly."""
    map_boundary, obstacles, start_point, end_point = simple_environment
    controller = RecedingHorizonController(
        map_boundary, obstacles, start_point, end_point,
        use_visualizer=False
    )
    
    # Check that basic attributes are set correctly
    assert np.array_equal(controller.start_point, np.array(start_point))
    assert np.array_equal(controller.end_point, np.array(end_point))
    assert controller.map_boundary == map_boundary
    assert controller.obstacles == obstacles
    assert len(controller.trajectory) == 1  # Should contain only start point initially
    assert np.array_equal(controller.trajectory[0], np.array(start_point))

def test_build_visibility_graph(controller):
    """Test the visibility graph construction."""
    graph, points = controller.graph, controller.points
    
    # Test graph has expected properties
    assert isinstance(graph, dict)
    assert len(points) > 0
    
    # Check that end point is in the graph
    assert tuple(controller.end_point) in graph
    
    # Verify all obstacle points are in the graph
    for obstacle in controller.obstacles:
        for point in obstacle:
            assert tuple(point) in graph

def test_dijkstra(controller):
    """Test the Dijkstra algorithm implementation."""
    cost_to_go = controller.cost_to_go
    
    # Cost to go from end point to itself should be 0
    assert cost_to_go[tuple(controller.end_point)] == 0
    
    # All other points should have finite costs
    for point in controller.points:
        if not np.array_equal(point, controller.end_point):
            assert cost_to_go[tuple(point)] < float('inf')

def test_solve_milp_returns_trajectory(controller):
    """Test that the MILP solver returns a valid trajectory."""
    # Solve from the start point
    trajectory = controller._solve_milp(controller.start_point)
    
    # Should return a numpy array
    assert isinstance(trajectory, np.ndarray)
    
    # First point should be the start point
    assert np.allclose(trajectory[0], controller.start_point)
    
    # Should have N+1 points
    assert len(trajectory) == controller.N + 1
    
    # All points should be 2D
    assert trajectory.shape[1] == 2

def test_boundary_constraints(controller):
    """Test that the trajectory stays within map boundaries."""
    # Get a trajectory
    trajectory = controller._solve_milp(controller.start_point)
    
    # Get map bounds
    boundary_points = np.array(controller.map_boundary)
    map_xmin, map_ymin = np.min(boundary_points, axis=0)
    map_xmax, map_ymax = np.max(boundary_points, axis=0)
    
    # Check all points are within boundaries (with margin)
    margin = controller.BOUNDARY_MARGIN
    for point in trajectory:
        assert point[0] >= map_xmin + margin
        assert point[0] <= map_xmax - margin
        assert point[1] >= map_ymin + margin
        assert point[1] <= map_ymax - margin

def test_obstacle_avoidance(controller):
    """Test that the trajectory avoids obstacles."""
    # Get a trajectory
    trajectory = controller._solve_milp(controller.start_point)
    
    # For each obstacle, check that no point is inside
    for obstacle in controller.obstacles:
        # Get bounding box of the obstacle
        obstacle_points = np.array(obstacle)
        xmin, ymin = np.min(obstacle_points, axis=0)
        xmax, ymax = np.max(obstacle_points, axis=0)
        
        # Check each trajectory point
        for point in trajectory:
            # If point is in the bounding box, it might be in the obstacle
            if (xmin <= point[0] <= xmax and ymin <= point[1] <= ymax):
                # Need more sophisticated check here for non-rectangular obstacles
                # For simplicity, we'll just use the bounding box
                assert False, f"Point {point} appears to be inside obstacle bounding box"

def test_execution_step(controller):
    """Test that execution steps move toward the goal."""
    # Get the initial distance to goal
    initial_distance = np.linalg.norm(controller.start_point - controller.end_point)
    
    # Solve for a path
    trajectory = controller._solve_milp(controller.start_point)
    
    # Check that the execution point (Ne steps ahead) is closer to the goal
    execution_point = trajectory[controller.Ne]
    new_distance = np.linalg.norm(execution_point - controller.end_point)
    
    assert new_distance < initial_distance, "Execution step should move closer to the goal"