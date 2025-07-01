import matplotlib.pyplot as plt
import pygame
import numpy as np
import time

def visualize_map(map_boundary, obstacles, graph, end_point, start_point=None):
    """Visualize the map, obstacles, and network."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot map boundary
    boundary_x, boundary_y = zip(*map_boundary + [map_boundary[0]])
    ax.plot(boundary_x, boundary_y, color='black', linewidth=1.5, label='Boundary')

    # Plot obstacles - single color for all obstacles
    for i, obstacle in enumerate(obstacles):
        obstacle_x, obstacle_y = zip(*obstacle + [obstacle[0]])
        ax.fill(obstacle_x, obstacle_y, color='lightgray', alpha=0.5)
        ax.plot(obstacle_x, obstacle_y, color='gray', linestyle='--')
    
    # Only add one legend entry for obstacles
    if obstacles:
        ax.plot([], [], color='gray', linestyle='--', label='Obstacles')

    # Plot graph edges
    if graph:
        for node, neighbors in graph.items():
            for neighbor in neighbors:
                ax.plot(
                    [node[0], neighbor[0]], [node[1], neighbor[1]], 
                    color='blue', alpha=0.5
                )
        # Add one legend entry for visibility graph
        ax.plot([], [], color='blue', alpha=0.7, label='Visibility Graph')

    # Plot end point
    ax.scatter(*end_point, color='red', s=100, label='Goal')
    
    # Plot start point if provided
    if start_point:
        ax.scatter(*start_point, color='green', s=100, label='Start')

    # Set equal aspect ratio for proper scaling
    ax.set_aspect('equal')
    
    # Move legend outside to the right
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.title("Map with Obstacles")
    plt.tight_layout()
    
    plt.show()

class PlannerVisualizer:
    def __init__(self, width=800, height=600, scale=25, background_color=(255, 255, 255)):
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
        self.clock = pygame.time.Clock()  # Initialize the clock
        
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
        
    def update(self, vehicle_pos, obstacles, actual_trajectory=None, predicted_trajectory=None, map_boundary=None, debug_info=None):
        """Update the visualization with new data"""
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            # Add keyboard event handling
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    self.running = False
                    print("Visualization manually terminated")
    
        if not self.running:
            return
        
        # Fill background
        self.screen.fill(self.background_color)
        
        # Draw map boundary if provided
        if map_boundary:
            self.draw_boundary(map_boundary)
        
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
        
        # Add exit message at the bottom of the screen
        font = pygame.font.SysFont(None, 24)  # Default font, size 24
        exit_msg = font.render("Press ESC or Q to exit", True, (255, 0, 0))  # Red text
        self.screen.blit(exit_msg, (10, self.height - 30))  # Position near bottom
    
        # Display debug information
        if debug_info:
            font = pygame.font.SysFont('Arial', 16)
            y_offset = 10
            for key, value in debug_info.items():
                text = f"{key}: {value}"
                text_surface = font.render(text, True, (0, 0, 0))
                self.screen.blit(text_surface, (10, y_offset))
                y_offset += 20
        
        # Update the display
        pygame.display.flip()
        
        # Limit framerate
        self.clock.tick(20)  # 20 FPS
        return True
        
    def close(self):
        """Close the visualization window"""
        pygame.quit()

    def draw_boundary(self, boundary, color=(100, 100, 100), width=2):
        """Draw the map boundary as a polygon"""
        if not boundary or len(boundary) < 3:
            return
            
        screen_points = [self.world_to_screen(p[0], p[1]) for p in boundary]
        pygame.draw.polygon(self.screen, color, screen_points, width)

