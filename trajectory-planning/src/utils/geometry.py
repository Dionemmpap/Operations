from shapely.geometry import LineString, Polygon, Point
import numpy as np

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

def find_tangents_to_circle(point, circle_center, radius):
    """
    Find tangent points from point to circle.
    
    Args:
        point: (x, y) point from which to draw tangents
        circle_center: (x, y) center of the circle
        radius: radius of the circle
        
    Returns:
        List of (x, y) tangent points on the circle
    """
    p = np.array(point)
    c = np.array(circle_center)
    r = radius
    
    # Vector from circle center to point
    pc = p - c
    dist = np.linalg.norm(pc)
    
    # If point is inside or on circle, no tangents possible
    if dist <= r:
        return []
    
    # Normalize pc vector
    pc_norm = pc / dist
    
    # Calculate perpendicular vector
    perp = np.array([-pc_norm[1], pc_norm[0]])
    
    # Calculate tangent points
    a = r * r / dist
    b = np.sqrt(r * r - a * a)
    
    # Two tangent points
    t1 = c + a * pc_norm + b * perp
    t2 = c + a * pc_norm - b * perp
    
    return [tuple(t1), tuple(t2)]

def is_valid_tangent(start_pos, tangent_point, target_pos, circle_center, turn_dir):
    """
    Check if tangent is compatible with turning direction.
    
    Args:
        start_pos: (x, y) starting position
        tangent_point: (x, y) tangent point on circle
        target_pos: (x, y) target position
        circle_center: (x, y) center of turning circle
        turn_dir: 'cw' or 'ccw' turning direction
    
    Returns:
        Boolean indicating if tangent is valid
    """
    # Vector from tangent point to circle center
    tc = np.array(circle_center) - np.array(tangent_point)
    
    # Vector from tangent point to target position
    tt = np.array(target_pos) - np.array(tangent_point)
    
    # Cross product to determine relative position
    cross = np.cross([tc[0], tc[1], 0], [tt[0], tt[1], 0])[2]
    
    # For clockwise, target should be on right side of tangent->center vector
    # For counter-clockwise, target should be on left side
    return (turn_dir == 'cw' and cross < 0) or (turn_dir == 'ccw' and cross > 0)

def calculate_arc_length(start_point, end_point, circle_center, radius, turn_dir):
    """
    Calculate length of arc on circle.
    
    Args:
        start_point: (x, y) starting point on circle
        end_point: (x, y) ending point on circle
        circle_center: (x, y) center of circle
        radius: radius of the circle
        turn_dir: 'cw' or 'ccw' turning direction
    
    Returns:
        Length of the arc
    """
    s = np.array(start_point)
    e = np.array(end_point)
    c = np.array(circle_center)
    
    # Vectors from center to points
    cs = s - c
    ce = e - c
    
    # Calculate angle between vectors
    cs_norm = cs / np.linalg.norm(cs)
    ce_norm = ce / np.linalg.norm(ce)
    dot = np.dot(cs_norm, ce_norm)
    angle = np.arccos(np.clip(dot, -1.0, 1.0))
    
    # Determine if we need the major or minor arc
    cross = np.cross([cs[0], cs[1], 0], [ce[0], ce[1], 0])[2]
    if (turn_dir == 'cw' and cross > 0) or (turn_dir == 'ccw' and cross < 0):
        angle = 2 * np.pi - angle
    
    # Arc length = radius * angle
    return radius * angle

def is_path_obstructed(start_pos, tangent_point, target_pos, circle_center, turn_dir, radius, obstacles, segments=8):
    """
    Check if path intersects obstacles.
    
    Args:
        start_pos: (x, y) starting position
        tangent_point: (x, y) tangent point on circle
        target_pos: (x, y) target position on circle
        circle_center: (x, y) center of turning circle
        turn_dir: 'cw' or 'ccw' turning direction
        radius: radius of the turning circle
        obstacles: list of obstacle polygons
        segments: number of segments to approximate arc
    
    Returns:
        Boolean indicating if path is obstructed
    """
    # Check straight line segment
    if is_path_blocked(start_pos, tangent_point, obstacles):
        return True
    
    # For arc, approximate with line segments
    arc_points = approximate_arc(tangent_point, target_pos, circle_center, radius, turn_dir, segments)
    
    # Check each segment
    for i in range(len(arc_points) - 1):
        if is_path_blocked(arc_points[i], arc_points[i+1], obstacles):
            return True
    
    return False

def approximate_arc(start_point, end_point, circle_center, radius, turn_dir, segments=8):
    """
    Approximate arc with line segments.
    
    Args:
        start_point: (x, y) starting point on circle
        end_point: (x, y) ending point on circle
        circle_center: (x, y) center of circle
        radius: radius of the circle
        turn_dir: 'cw' or 'ccw' turning direction
        segments: number of segments to use
        
    Returns:
        List of points approximating the arc
    """
    s = np.array(start_point)
    e = np.array(end_point)
    c = np.array(circle_center)
    
    # Vectors from center to points
    cs = s - c
    ce = e - c
    r = radius  # Using the provided radius
    
    # Calculate angle between vectors
    angle = calculate_arc_angle(cs, ce, turn_dir)
    
    # Generate points along the arc
    points = [start_point]
    for i in range(1, segments):
        t = i / segments
        current_angle = angle * t
        
        # Rotate the start vector
        if turn_dir == 'cw':
            current_angle = -current_angle
        
        # Rotation matrix
        rot = np.array([
            [np.cos(current_angle), -np.sin(current_angle)],
            [np.sin(current_angle), np.cos(current_angle)]
        ])
        
        # Apply rotation to cs
        cs_norm = cs / np.linalg.norm(cs)
        current_vec = np.dot(rot, cs_norm) * r
        current_point = c + current_vec
        
        points.append(tuple(current_point))
    
    points.append(end_point)
    return points

def calculate_arc_angle(v1, v2, turn_dir):
    """
    Calculate angle between vectors for arc.
    
    Args:
        v1: First vector [x, y]
        v2: Second vector [x, y]
        turn_dir: 'cw' or 'ccw' turning direction
        
    Returns:
        Angle in radians
    """
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)
    
    # Calculate dot product and angle
    dot = np.dot(v1_norm, v2_norm)
    angle = np.arccos(np.clip(dot, -1.0, 1.0))
    
    # Determine major or minor arc
    cross = np.cross([v1[0], v1[1], 0], [v2[0], v2[1], 0])[2]
    if (turn_dir == 'cw' and cross > 0) or (turn_dir == 'ccw' and cross < 0):
        angle = 2 * np.pi - angle
    
    return angle