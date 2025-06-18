import matplotlib.pyplot as plt
import pygame
import numpy as np
import time

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

class PlannerVisualizer:
    def __init__(self, width=800, height=600, scale=10, background_color=(255, 255, 255)):
        """
        Initialize the pygame visualizer for a receding horizon planner.
        
        Args:
            width: Window width in pixels
            height: Window height in pixels
            scale: Pixels per unit (for scaling coordinates)
            background_color: RGB background color
        """
        pygame.init()
        self.width = width
        self.height = height
        self.scale = scale
        self.background_color = background_color
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Receding Horizon Planner Visualization")
        self.running = True
        
        # We'll calculate these dynamically
        self.center_x = 0
        self.center_y = 0
        self.min_x = float('inf')
        self.max_x = float('-inf')
        self.min_y = float('inf')
        self.max_y = float('-inf')
        
    def set_world_bounds(self, obstacles, start_point, end_point):
        """Calculate the world bounds to ensure everything is visible"""
        # Reset bounds
        self.min_x = float('inf')
        self.max_x = float('-inf')
        self.min_y = float('inf')
        self.max_y = float('-inf')
        
        # Check start and end points
        self.update_bounds(start_point[0], start_point[1])
        self.update_bounds(end_point[0], end_point[1])
        
        # Check all obstacle vertices
        for obs in obstacles:
            if isinstance(obs, list):
                for point in obs:
                    self.update_bounds(point[0], point[1])
            elif isinstance(obs, tuple) and len(obs) >= 3:
                # For circular obstacles, check the extremes
                self.update_bounds(obs[0] - obs[2], obs[1] - obs[2])  # Bottom left
                self.update_bounds(obs[0] + obs[2], obs[1] + obs[2])  # Top right
        
        # Add some padding (10%)
        width = self.max_x - self.min_x
        height = self.max_y - self.min_y
        padding_x = width * 0.1
        padding_y = height * 0.1
        
        self.min_x -= padding_x
        self.max_x += padding_x
        self.min_y -= padding_y
        self.max_y += padding_y
        
        # Calculate the world center
        self.center_x = (self.min_x + self.max_x) / 2
        self.center_y = (self.min_y + self.max_y) / 2
        
        # Calculate appropriate scale
        width = self.max_x - self.min_x
        height = self.max_y - self.min_y
        
        if width > 0 and height > 0:
            scale_x = self.width / width
            scale_y = self.height / height
            self.scale = min(scale_x, scale_y) * 0.9  # 90% to leave margin
        
        print(f"World bounds: x={self.min_x:.1f} to {self.max_x:.1f}, y={self.min_y:.1f} to {self.max_y:.1f}")
        print(f"Center: ({self.center_x:.1f}, {self.center_y:.1f}), Scale: {self.scale:.1f}")
    
    def update_bounds(self, x, y):
        """Update the min/max bounds with a new point"""
        self.min_x = min(self.min_x, x)
        self.max_x = max(self.max_x, x)
        self.min_y = min(self.min_y, y)
        self.max_y = max(self.max_y, y)
        
    def world_to_screen(self, x, y):
        """Convert world coordinates to screen coordinates"""
        # Center the view on the world center
        screen_x = int((x - self.center_x) * self.scale + self.width / 2)
        screen_y = int((self.center_y - y) * self.scale + self.height / 2)
        return screen_x, screen_y
        
    def draw_vehicle(self, x, y, heading=0, color=(0, 0, 255), size=5):
        """Draw the vehicle at the given position"""
        pos = self.world_to_screen(x, y)
        pygame.draw.circle(self.screen, color, pos, size)
        
        # Draw a line indicating heading
        end_x = pos[0] + int(size * 2 * np.cos(heading))
        end_y = pos[1] - int(size * 2 * np.sin(heading))
        pygame.draw.line(self.screen, color, pos, (end_x, end_y), 2)
        
    def draw_obstacle(self, obstacle, color=(255, 0, 0)):
        """
        Draw an obstacle - can be a polygon or a circle
        
        Args:
            obstacle: Either (x, y, radius) for a circle or a list of (x, y) points for a polygon
            color: RGB color tuple
        """
        if isinstance(obstacle, tuple) and len(obstacle) >= 3:
            # It's a circle (x, y, radius)
            pos = self.world_to_screen(obstacle[0], obstacle[1])
            radius = int(obstacle[2] * self.scale)
            
            # Use custom color if provided
            if len(obstacle) >= 4 and isinstance(obstacle[3], tuple):
                color = obstacle[3]
                
            pygame.draw.circle(self.screen, color, pos, radius)
        elif isinstance(obstacle, list) and len(obstacle) > 2:
            # It's a polygon - list of points
            screen_points = [self.world_to_screen(p[0], p[1]) for p in obstacle]
            pygame.draw.polygon(self.screen, color, screen_points)
        
    def draw_trajectory(self, points, color=(0, 255, 0), width=2):
        """Draw a trajectory from a list of (x, y) points"""
        if len(points) < 2:
            return
            
        screen_points = [self.world_to_screen(x, y) for x, y in points]
        pygame.draw.lines(self.screen, color, False, screen_points, width)
        
    def draw_predicted_trajectory(self, points, color=(0, 200, 200), width=1):
        """Draw the predicted trajectory from the planner"""
        if len(points) < 2:
            return
            
        screen_points = [self.world_to_screen(x, y) for x, y in points]
        pygame.draw.lines(self.screen, color, False, screen_points, width)
        
        # Draw points at each prediction step
        for point in screen_points:
            pygame.draw.circle(self.screen, color, point, 2)
        
    def update(self, vehicle_pos, obstacles, actual_trajectory=None, predicted_trajectory=None):
        """
        Update the visualization with new data
        
        Args:
            vehicle_pos: (x, y, heading) of the vehicle
            obstacles: List of obstacles (either circles or polygons)
            actual_trajectory: List of (x, y) points showing past trajectory
            predicted_trajectory: List of (x, y) points showing predicted future trajectory
        """
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                
        if not self.running:
            return False
                
        # Clear screen
        self.screen.fill(self.background_color)
        
        # Draw obstacles
        for obs in obstacles:
            self.draw_obstacle(obs)
        
        # Draw trajectories
        if actual_trajectory:
            self.draw_trajectory(actual_trajectory)
            
        if predicted_trajectory:
            self.draw_predicted_trajectory(predicted_trajectory)
            
        # Draw vehicle
        self.draw_vehicle(vehicle_pos[0], vehicle_pos[1], 
                         vehicle_pos[2] if len(vehicle_pos) > 2 else 0)
        
        # Update display
        pygame.display.flip()
        # Small delay to ensure visualization updates
        time.sleep(0.05)
        return True
        
    def close(self):
        """Close the visualization window"""
        pygame.quit()


# Example usage:
def demo_visualizer():
    # Create the visualizer
    viz = PlannerVisualizer()
    
    # Define some obstacles
    obstacles = [
        (-5, -5, 1),
        (5, 5, 1),
        (10, 0, 2),
        (-2, 8, 1.5)
    ]
    
    # Simulate a vehicle moving along a trajectory
    actual_trajectory = []
    t = 0
    
    while viz.running and t < 1000:
        # Simulate vehicle motion (circle in this example)
        x = 5 * np.cos(t * 0.05)
        y = 5 * np.sin(t * 0.05)
        heading = t * 0.05 + np.pi/2
        
        # Save the actual trajectory
        actual_trajectory.append((x, y))
        if len(actual_trajectory) > 50:  # Keep only recent history
            actual_trajectory.pop(0)
            
        # Simulate a predicted trajectory (just a simple extrapolation)
        predicted_trajectory = []
        for i in range(1, 20):
            pred_x = x + i * 0.5 * np.cos(heading)
            pred_y = y + i * 0.5 * np.sin(heading)
            predicted_trajectory.append((pred_x, pred_y))
            
        # Update the visualization
        if not viz.update((x, y, heading), obstacles, actual_trajectory, predicted_trajectory):
            break
            
        # Small delay to simulate computation time
        time.sleep(0.05)
        t += 1
        
    viz.close()

if __name__ == "__main__":
    demo_visualizer()