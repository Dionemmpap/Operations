# filepath: /trajectory-planning/trajectory-planning/src/planners/milp_planner.py

import numpy as np
import gurobipy as gp
from gurobipy import GRB
from .base_planner import TrajectoryDesignBase
from utils.geometry import is_path_blocked

class MILPTrajectoryPlanner(TrajectoryDesignBase):
    """MILP-based trajectory planner with dynamic feasibility constraints."""
    
    def plan_trajectory(self, horizon=5, max_iterations=20):
        """Run the trajectory planning algorithm with MILP."""
        print("Planning trajectory with MILP approach...")
        self.receding_horizon_milp(horizon, max_iterations)
        return self.trajectory
    
    def receding_horizon_milp(self, horizon=5, max_iterations=20):
        """Plan trajectory with receding horizon control using MILP."""
        current_state = self.state
        self.trajectory = [current_state[:2]]
        iteration = 0
        
        while np.linalg.norm(current_state[:2] - self.end_point) > 0.5:
            iteration += 1
            if iteration > max_iterations:
                print(f"Stopping after {max_iterations} iterations")
                break
            
            print(f"MILP Iteration {iteration}: Planning from {current_state[:2]}")
            
            # Plan trajectory with MILP
            trajectory = self.plan_trajectory_milp(current_state, horizon)
            
            if trajectory is None or len(trajectory) < 2:
                print("MILP planning failed, trying systematic exploration...")
                next_state = self.systematic_exploration(current_state)
                if next_state is None:
                    print("Systematic exploration failed. Stopping.")
                    break
                current_state = next_state
            else:
                # Take the next state from the trajectory
                next_state = trajectory[1]  # Skip the first state which is the current state
                current_state = next_state
            
            print(f"  Moving to {current_state[:2]}, " 
                  f"distance: {np.linalg.norm(current_state[:2] - self.end_point):.4f}")
            
            self.trajectory.append(current_state[:2])
    
    def plan_trajectory_milp(self, current_state, horizon=5):
        """Plan a trajectory using Mixed Integer Linear Programming (MILP)."""
        try:
            # Create a new model
            model = gp.Model("trajectory_planning")
            model.setParam('OutputFlag', 0)  # Suppress output except for final solution
            
            # Time step for discretization
            dt = self.tau
            
            # Maximum velocity and control bounds
            max_velocity = 2.0
            max_accel = 0.5
            
            # Create variables for each time step - CHANGED: define individually
            x = {}
            y = {}
            v = {} 
            for t in range(horizon+1):
                x[t] = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name=f"x_{t}")
                y[t] = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name=f"y_{t}")
                v[t] = model.addVar(lb=0, ub=max_velocity, name=f"v_{t}")
        
            # Control inputs
            a = {}
            for t in range(horizon):
                a[t] = model.addVar(lb=-max_accel, ub=max_accel, name=f"a_{t}")
        
            # Direction variables
            dx = {}
            dy = {}
            for t in range(horizon):
                dx[t] = model.addVar(lb=-1, ub=1, name=f"dx_{t}")
                dy[t] = model.addVar(lb=-1, ub=1, name=f"dy_{t}")
        
            # CHANGED: Set initial state using constraints instead of direct assignment
            model.addConstr(x[0] == current_state[0])
            model.addConstr(y[0] == current_state[1])
            model.addConstr(v[0] == current_state[3])
        
            # Direction discretization
            for t in range(horizon):
                # Add 8 binary variables for direction segments
                segs = {}
                for i in range(8):
                    segs[i] = model.addVar(vtype=GRB.BINARY, name=f"dir_seg_{t}_{i}")
            
                model.addConstr(gp.quicksum(segs[i] for i in range(8)) == 1)  # Exactly one segment active
            
                # Define 8 directions (unit vectors)
                directions = [
                    (1, 0), (0.7071, 0.7071), (0, 1), (-0.7071, 0.7071),
                    (-1, 0), (-0.7071, -0.7071), (0, -1), (0.7071, -0.7071)
                ]
            
                # Constrain dx, dy to be a convex combination of adjacent directions
                model.addConstr(dx[t] == gp.quicksum(directions[i][0] * segs[i] for i in range(8)))
                model.addConstr(dy[t] == gp.quicksum(directions[i][1] * segs[i] for i in range(8)))
    
            # Linearized dynamics
            for t in range(horizon):
                model.addConstr(x[t+1] == x[t] + v[t] * dt * dx[t])
                model.addConstr(y[t+1] == y[t] + v[t] * dt * dy[t])
                model.addConstr(v[t+1] == v[t] + a[t] * dt)
        
            # Obstacle avoidance constraints
            for t in range(1, horizon+1):
                for obs_idx, obstacle in enumerate(self.obstacles):
                    # Add binary variables for each edge of the obstacle
                    binary_vars = []
                    
                    for i in range(len(obstacle)):
                        p1 = obstacle[i]
                        p2 = obstacle[(i+1) % len(obstacle)]
                        
                        # Vector normal to the edge (pointing outward)
                        edge_vec = np.array([p2[0] - p1[0], p2[1] - p1[1]])
                        normal = np.array([-edge_vec[1], edge_vec[0]])  # Rotate 90 degrees
                        
                        # Ensure normal points outward
                        center = np.mean(obstacle, axis=0)
                        edge_midpoint = np.array([(p1[0] + p2[0])/2, (p1[1] + p2[1])/2])
                        outward = edge_midpoint - center
                        
                        if np.dot(normal, outward) < 0:
                            normal = -normal  # Flip if pointing inward
                        
                        # Normalize
                        norm = np.linalg.norm(normal)
                        if norm > 0:
                            normal = normal / norm
                        
                        # Create binary variable for this edge
                        b = model.addVar(vtype=GRB.BINARY, name=f"b_{t}_{obs_idx}_{i}")
                        binary_vars.append(b)
                        
                        # Set up the constraint: point must be on outside of edge
                        M = 100  # Big-M value
                        model.addConstr(
                            (x[t] - p1[0])*normal[0] + (y[t] - p1[1])*normal[1] >= -M*(1-b)
                        )
                    
                    # Must satisfy at least one edge constraint
                    model.addConstr(gp.quicksum(binary_vars) >= 1)
        
            # Objective function
            goal_x, goal_y = self.end_point
            
            # Distance to goal at final state
            final_dist = (x[horizon] - goal_x)*(x[horizon] - goal_x) + (y[horizon] - goal_y)*(y[horizon] - goal_y)
            
            # Approach goal at each step
            approach_goal = gp.quicksum(
                ((x[t] - goal_x)*(x[t] - goal_x) + (y[t] - goal_y)*(y[t] - goal_y)) for t in range(horizon+1)
            )
            
            # Control effort
            control_effort = gp.quicksum(a[t]*a[t] for t in range(horizon))
            
            # Objective weights
            w_final = 10.0
            w_approach = 1.0
            w_control = 0.1
            
            model.setObjective(
                w_final * final_dist + 
                w_approach * approach_goal + 
                w_control * control_effort, 
                GRB.MINIMIZE
            )
            
            # Solve the optimization problem
            model.optimize()
            
            if model.status == GRB.OPTIMAL:
                # Extract the optimal trajectory
                trajectory = []
                for t in range(horizon+1):
                    # Calculate theta from dx, dy for t > 0
                    if t > 0:
                        theta_val = np.arctan2(dy[t-1].X, dx[t-1].X)
                    else:
                        theta_val = current_state[2]
                    
                    state = np.array([x[t].X, y[t].X, theta_val, v[t].X])
                    trajectory.append(state)
                
                return trajectory
            else:
                print(f"Optimization failed with status {model.status}")
                return None
                
        except gp.GurobiError as e:
            print(f"Gurobi error: {e}")
            return None
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error in MILP planning: {e}")
            return None
    
    def systematic_exploration(self, current_state, step_size=0.2):
        """
        Systematic exploration based on visibility graph and potential field.
        Uses informed directional sampling rather than random exploration.
        """
        # Get current position and heading
        current_pos = current_state[:2]
        current_heading = current_state[2]
        
        # 1. First try: Use visibility graph to find a visible node that's closer to the goal
        # Temporarily add current position to graph
        current_pos_tuple = tuple(current_pos)
        visible_nodes = []
        
        # Find all visible nodes from current position
        for point in self.points:
            point_tuple = tuple(point)
            if not is_path_blocked(current_pos, point, self.obstacles):
                visible_nodes.append(point_tuple)
        
        # If there are visible nodes, find the one that minimizes distance to goal
        if visible_nodes:
            closest_node = min(visible_nodes, 
                              key=lambda n: np.linalg.norm(np.array(n) - np.array(self.end_point)))
            
            # Move toward this node
            direction = np.array(closest_node) - current_pos
            dist = np.linalg.norm(direction)
            
            if dist > step_size:
                direction = direction / dist * step_size
            
            # Create a new state moving toward this node
            new_heading = np.arctan2(direction[1], direction[0])
            new_pos = current_pos + direction
            new_velocity = min(current_state[3] + 0.1, 2.0)  # Slightly increase velocity
            
            return np.array([new_pos[0], new_pos[1], new_heading, new_velocity])
        
        # 2. Second try: Use potential field approach
        # Calculate repulsive forces from obstacles
        repulsive_force = np.zeros(2)
        for obstacle in self.obstacles:
            # Find closest point on obstacle
            min_dist = float('inf')
            closest_point = None
            
            for i in range(len(obstacle)):
                p1 = np.array(obstacle[i])
                p2 = np.array(obstacle[(i+1) % len(obstacle)])
                
                # Find closest point on line segment to current position
                segment = p2 - p1
                segment_length = np.linalg.norm(segment)
                if segment_length > 0:
                    segment_unit = segment / segment_length
                    # Project current position onto line segment
                    t = np.dot(current_pos - p1, segment_unit)
                    t = max(0, min(segment_length, t))  # Clamp to segment
                    closest = p1 + t * segment_unit
                    
                    dist = np.linalg.norm(current_pos - closest)
                    if dist < min_dist:
                        min_dist = dist
                        closest_point = closest
            
            if closest_point is not None:
                # Repulsive force inversely proportional to distance
                if min_dist < 1.0:  # Only consider obstacles within a certain range
                    force_magnitude = 1.0 / (min_dist * min_dist + 0.1)  # Avoid division by zero
                    force_direction = current_pos - closest_point
                    if np.linalg.norm(force_direction) > 0:
                        force_direction = force_direction / np.linalg.norm(force_direction)
                    repulsive_force += force_magnitude * force_direction
        
        # Attractive force toward goal
        goal_direction = np.array(self.end_point) - current_pos
        goal_distance = np.linalg.norm(goal_direction)
        if goal_distance > 0:
            goal_direction = goal_direction / goal_distance
        
        # Combine forces
        attractive_weight = 1.0
        repulsive_weight = 1.5
        combined_force = attractive_weight * goal_direction + repulsive_weight * repulsive_force
        
        # Normalize the combined force
        force_magnitude = np.linalg.norm(combined_force)
        if force_magnitude > 0:
            combined_force = combined_force / force_magnitude * step_size
            
            # Create new state based on combined force
            new_heading = np.arctan2(combined_force[1], combined_force[0])
            new_pos = current_pos + combined_force
            
            # Check if the new position is valid
            if not is_path_blocked(current_pos, new_pos, self.obstacles):
                new_velocity = min(current_state[3] + 0.1, 2.0)
                return np.array([new_pos[0], new_pos[1], new_heading, new_velocity])
        
        # If all else fails, try moving in the direction of the current heading
        forward_direction = np.array([np.cos(current_heading), np.sin(current_heading)])
        new_pos = current_pos + forward_direction * step_size
        
        if not is_path_blocked(current_pos, new_pos, self.obstacles):
            return np.array([new_pos[0], new_pos[1], current_heading, current_state[3]])
        
        # No valid move found
        return None