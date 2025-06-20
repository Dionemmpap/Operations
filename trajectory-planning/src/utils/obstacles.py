import numpy as np
from shapely.geometry import Polygon, MultiPolygon


def get_obstacles(map_boundary, num_obstacles):
    obstacles = []
    for _ in range(num_obstacles):
        x1 = np.random.uniform(0.15*map_boundary[1][0], 0.75*map_boundary[1][0])
        y1 = np.random.uniform(0.15*map_boundary[2][1], 0.75*map_boundary[2][1])
        width = np.random.uniform(0.1*map_boundary[1][0], 0.25*map_boundary[1][0])
        height = np.random.uniform(0.1*map_boundary[2][1], 0.25*map_boundary[2][1])
        x2 = x1 + width
        y2 = y1
        x3 = x2
        y3 = y1 + height
        x4 = x1
        y4 = y3
        obstacles.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])


    return obstacles

def merge_intersecting_obstacles(obstacles):
    """Merge intersecting obstacles into a single larger obstacle."""
    # Convert obstacles to shapely Polygons
    polygons = [Polygon(obstacle) for obstacle in obstacles]

    # Merge all polygons using buffer(0)
    merged = MultiPolygon(polygons).buffer(0)

    # Check if merged result is a single Polygon
    if isinstance(merged, Polygon):
        return [list(merged.exterior.coords[:-1])]

    # Check if merged result is a MultiPolygon
    elif isinstance(merged, MultiPolygon):
        return [list(poly.exterior.coords[:-1]) for poly in merged.geoms]

    # If no merging occurs (fallback)
    return obstacles