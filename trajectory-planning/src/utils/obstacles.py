import numpy as np
from shapely.geometry import Polygon, MultiPolygon
import math


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


class node(object):
    """A class to represent a node in the trajectory planning tree."""
    
    def __init__(self, pos, vel=None, cost_to_go=float('inf')):
        self.pos = pos          # Position as a tuple (x, y)
        self.vel = vel          # Velocity as a tuple (vx, vy)
        self.cost_to_go = cost_to_go # Cost to go from this node to the goal
        self.circle_cw = None
        self.circle_ccw = None
        self.id = None
        self.successor = None
        self.path_geometry = None  # Add this to store path info

    def __repr__(self):
        return (f"Node(id={self.id}, pos={self.pos}, vel={self.vel}, "
                f"cost_to_go={self.cost_to_go:.2f})")
        
    def calc_circle_centers(self, rho, vmax):
        """
        Calculates the centers of the kinodynamic turning circles for this node.

        This method implements Equations 3.10 and 3.11 from the source paper. 
        It computes the centers of the two circles (clockwise and counter-clockwise)
        of radius `rho` to which the node's velocity vector `self.vel` is tangent
        at the node's position `self.pos`. 

        Args:
            rho (float): The vehicle's minimum turning radius.
            vmax (float): The vehicle's maximum speed.
        """
        # Ensure velocity is not a zero vector to avoid division by zero
        # and handle cases where the node is not yet assigned a velocity.
        if self.vel is None or (self.vel[0] == 0 and self.vel[1] == 0):
            self.circle_cw = None
            self.circle_ccw = None
            return

        # The paper assumes the velocity magnitude is vmax.
        # The perpendicular vector is scaled by rho/vmax.
        # Let vel = (vx, vy). The two perpendicular vectors are (vy, -vx) and (-vy, vx).
        
        vx, vy = self.vel
        
        # --- Clockwise Circle Center (Equation 3.10) --- 
        # Uses the perpendicular vector [vy, -vx]
        offset_cw_x = (rho / vmax) * vy
        offset_cw_y = (rho / vmax) * (-vx)
        self.circle_cw = (self.pos[0] + offset_cw_x, self.pos[1] + offset_cw_y)

        # --- Counter-Clockwise Circle Center (Equation 3.11) --- 
        # Uses the perpendicular vector [-vy, vx]
        offset_ccw_x = (rho / vmax) * (-vy)
        offset_ccw_y = (rho / vmax) * vx
        self.circle_ccw = (self.pos[0] + offset_ccw_x, self.pos[1] + offset_ccw_y)

