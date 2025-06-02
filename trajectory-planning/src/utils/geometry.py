from shapely.geometry import LineString, Polygon, Point

def is_path_blocked(point1, point2, obstacles):
    """Check if the straight line between two points intersects or lies within any obstacle."""
    line = LineString([point1, point2])
    for obstacle in obstacles:
        polygon = Polygon(obstacle)
        if line.crosses(polygon) or line.within(polygon):
            return True
    return False

def is_point_on_boundary(point, map_boundary):
    """Check if a point lies on the map boundary."""
    # Check if point matches any boundary point
    if point in map_boundary:
        return True
    
    # Check if point lies on any boundary edge
    for i in range(len(map_boundary)):
        p1 = map_boundary[i]
        p2 = map_boundary[(i + 1) % len(map_boundary)]
        
        # Create line segment
        line = LineString([p1, p2])
        
        # Check if point lies on the line segment
        if line.distance(Point(point)) < 1e-6:
            return True
            
    return False