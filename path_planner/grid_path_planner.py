from path_finding import PathPlanner
import math
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

class GridPathPlanner(PathPlanner):
    def __init__(self, grid_resolution=0.4, target_radius=0.5, obstacle_size=0.1, robot_radius=0.2):
        """
        Initialize the GridPathPlanner considering the robot's radius.
        """
        super().__init__(grid_resolution, robot_radius=robot_radius, target_radius=target_radius, obstacle_size=obstacle_size)
        self.motion_model = self.define_motion_model()

    @staticmethod
    def heuristic(node1, node2):
        """
        Manhattan distance heuristic.
        """
        return abs(node1.x - node2.x) + abs(node1.y - node2.y)

    def define_motion_model(self):
        """
        Define the motion model to restrict to grid-based straight-line movement.
        """
        step_size = 0.2  # Define step size for grid movement (horizontal/vertical)
        return [[step_size, 0, 1], [0, step_size, 1], [-step_size, 0, 1], [0, -step_size, 1]]

    def find_turning_points(self, path_x, path_y):
        """
        Identify turning points in the path where the robot changes direction.
        Args:
            path_x: List of x coordinates of the path.
            path_y: List of y coordinates of the path.

        Returns:
            turning_x, turning_y: Lists of x and y coordinates of turning points.
        """
        turning_x, turning_y = [], []

        for i in range(1, len(path_x) - 1):
            prev_dx = path_x[i] - path_x[i - 1]
            prev_dy = path_y[i] - path_y[i - 1]
            next_dx = path_x[i + 1] - path_x[i]
            next_dy = path_y[i + 1] - path_y[i]

            # Check if the direction changes
            if (prev_dx != next_dx) or (prev_dy != next_dy):
                # If the direction changes, it's a turning point
                turning_x.append(path_x[i])
                turning_y.append(path_y[i])

        return turning_x, turning_y

    def simplify_path(self, path_x, path_y):
        """
        Simplify the path by removing unnecessary waypoints that lie on the same line.
        This reduces the zigzag effect.
        """
        if len(path_x) < 3:  # Path is already simple enough
            return path_x, path_y
        
        simplified_x, simplified_y = [path_x[0]], [path_y[0]]  # Start with the first point
        
        for i in range(1, len(path_x) - 1):
            prev_x, prev_y = simplified_x[-1], simplified_y[-1]
            cur_x, cur_y = path_x[i], path_y[i]
            next_x, next_y = path_x[i + 1], path_y[i + 1]

            # Check if the current point is redundant (lies in a straight line)
            if not self.check_collision(prev_x, prev_y, next_x, next_y):
                # Skip the current point if there is no obstacle in between
                continue

            # Add the current point to the simplified path
            simplified_x.append(cur_x)
            simplified_y.append(cur_y)

        # Add the last point
        simplified_x.append(path_x[-1])
        simplified_y.append(path_y[-1])

        return simplified_x, simplified_y

    def smooth_path(self, path_x, path_y, num_points=100):
        """
        Smooth the path using interpolation to create intermediate points.
        Args:
            path_x: List of x coordinates of the path.
            path_y: List of y coordinates of the path.
            num_points: Number of points for the smoothed path.

        Returns:
            smoothed_x, smoothed_y: Interpolated and smoothed x and y coordinates.
        """
        # Ensure there are at least 2 points to interpolate
        if len(path_x) < 2:
            return path_x, path_y
        
        # Calculate the cumulative distance along the path
        distance = np.cumsum(np.sqrt(np.ediff1d(path_x, to_begin=0)**2 + np.ediff1d(path_y, to_begin=0)**2))
        distance = distance / distance[-1]  # Normalize distance
        
        # Interpolation functions for x and y
        fx = interp1d(distance, path_x, kind='linear')
        fy = interp1d(distance, path_y, kind='linear')
        
        # Create equally spaced points along the path
        alpha = np.linspace(0, 1, num_points)
        smoothed_x = fx(alpha)
        smoothed_y = fy(alpha)
        
        return smoothed_x, smoothed_y

    def plan_path(self, start_x, start_y, goal_x, goal_y):
        """
        A* pathfinding algorithm for movement along grid lines (no diagonals).
        The robot will stop once it's within the specified proximity range of the goal.
        """
        start_time = time.time()

        # Initialize the start and goal nodes
        start_node = self.Node(self.get_xy_index(start_x, self.grid_min_x),
                            self.get_xy_index(start_y, self.grid_min_y), 0.0, -1)
        goal_node = self.Node(self.get_xy_index(goal_x, self.grid_min_x),
                            self.get_xy_index(goal_y, self.grid_min_y), 0.0, -1)

        open_set, closed_set = dict(), dict()
        open_set[self.get_node_index(start_node)] = start_node

        while True:
            if len(open_set) == 0:
                raise Exception("No path found.")

            # Get the node with the lowest cost + heuristic
            current_id = min(open_set, key=lambda o: open_set[o].cost + self.heuristic(goal_node, open_set[o]))
            current_node = open_set[current_id]

            # Check if the robot is within the proximity of the target
            if self.within_target_range(current_node, goal_node):
                print(f"Robot reached within {self.target_radius} units of the goal. Stopping.")
                goal_node.parent_idx = current_node.parent_idx
                goal_node.cost = current_node.cost
                break

            # Remove the current node from open set and add to closed set
            del open_set[current_id]
            closed_set[current_id] = current_node

            # Explore neighboring nodes
            for i, _ in enumerate(self.motion_model):
                new_node = self.Node(current_node.x + self.motion_model[i][0],
                                    current_node.y + self.motion_model[i][1],
                                    current_node.cost + self.motion_model[i][2], current_id)
                node_id = self.get_node_index(new_node)

                # Check if node is valid (collision-free)
                if not self.is_node_valid(new_node):
                    continue

                if node_id in closed_set:
                    continue

                # Update open set with the new node
                if node_id not in open_set:
                    open_set[node_id] = new_node
                else:
                    if open_set[node_id].cost > new_node.cost:
                        open_set[node_id] = new_node

        # Extract the final path
        path_x, path_y = self.extract_final_path(goal_node, closed_set)
        
        # Identify turning points in the path
        turning_x, turning_y = self.find_turning_points(path_x, path_y)

        # Simplify the path to reduce unnecessary waypoints
        simplified_x, simplified_y = self.simplify_path(turning_x, turning_y)

        # Apply smoothing to the simplified path
        smoothed_x, smoothed_y = self.smooth_path(simplified_x, simplified_y, num_points=200)

        end_time = time.time()
        print(f"Navigation Time: {end_time - start_time:.2f}s")
        
        return smoothed_x, smoothed_y

    def within_target_range(self, current_node, goal_node):
        """
        Check if the current node is within the proximity of the goal node based on the target_radius.
        """
        current_pos_x = self.get_grid_position(current_node.x, self.grid_min_x)
        current_pos_y = self.get_grid_position(current_node.y, self.grid_min_y)
        goal_pos_x = self.get_grid_position(goal_node.x, self.grid_min_x)
        goal_pos_y = self.get_grid_position(goal_node.y, self.grid_min_y)

        distance = math.hypot(current_pos_x - goal_pos_x, current_pos_y - goal_pos_y)
        return distance <= self.target_radius  # Proceed to next goal if within range
    
    def extract_final_path(self, goal_node, closed_set):
        """
        Extracts the final path from the goal to the start by following the parent links.
        """
        # Start with goal node
        path_x, path_y = [self.get_grid_position(goal_node.x, self.grid_min_x)], \
                        [self.get_grid_position(goal_node.y, self.grid_min_y)]
        parent_idx = goal_node.parent_idx

        # Traverse back to the start node
        while parent_idx != -1:
            node = closed_set[parent_idx]
            path_x.append(self.get_grid_position(node.x, self.grid_min_x))
            path_y.append(self.get_grid_position(node.y, self.grid_min_y))
            parent_idx = node.parent_idx

        # Return the path without the starting node (i.e., path from the second point onward)
        return path_x[1:], path_y[1:]

    def check_collision(self, x1, y1, x2, y2):
        """
        Check if there is a collision (obstacle) in the path between two points (x1, y1) and (x2, y2).
        Uses Bresenham's line algorithm to check for obstacles.
        """
        x1_idx = self.get_xy_index(x1, self.grid_min_x)
        y1_idx = self.get_xy_index(y1, self.grid_min_y)
        x2_idx = self.get_xy_index(x2, self.grid_min_x)
        y2_idx = self.get_xy_index(y2, self.grid_min_y)

        # Bresenham's line algorithm
        points = self.bresenham(x1_idx, y1_idx, x2_idx, y2_idx)
        for x_idx, y_idx in points:
            if self.obstacle_grid[x_idx][y_idx]:
                return True  # Collision detected
        return False  # No collision

    def bresenham(self, x1, y1, x2, y2):
        """
        Bresenham's line algorithm to determine grid points between two coordinates.
        """
        points = []
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        while True:
            points.append((x1, y1))
            if x1 == x2 and y1 == y2:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy
        return points


    def plan_path_based_mode(self, start_x, start_y, goal_x, goal_y):
        """
        Plan the path and return the x, y coordinates for the path.
        If the distance between two consecutive points exceeds the threshold,
        intermediate points will be added to ensure reliable movement.
        If the distance between two consecutive points is too small, those points
        will be skipped to avoid unnecessary noise.
        """
        min_distance = 0.2  # Minimum distance to avoid noise
        max_distance = 0.2  # Maximum distance before adding intermediate points

        # Get the planned path (x and y coordinates)
        path_x, path_y = self.plan_path(start_x, start_y, goal_x, goal_y)
        print(f"path_x: {path_x}, path_y: {path_y}")

        # Lists to store the new path with intermediate points and skipped small points
        new_path_x, new_path_y = [path_x[0]], [path_y[0]]  # Start with the first point

        for i in range(1, len(path_x)):
            # Calculate the distance between consecutive points
            dx = path_x[i] - path_x[i - 1]
            dy = path_y[i] - path_y[i - 1]
            distance = math.hypot(dx, dy)

            # If the distance is less than the minimum threshold, skip the point (noise reduction)
            if distance < min_distance:
                continue

            # If the distance exceeds the maximum threshold, add intermediate points
            if distance > max_distance:
                # Calculate the number of intermediate points to add
                num_intermediate_points = int(distance / max_distance)
                for j in range(1, num_intermediate_points + 1):
                    # Interpolate intermediate points
                    intermediate_x = path_x[i - 1] + (dx * j / (num_intermediate_points + 1))
                    intermediate_y = path_y[i - 1] + (dy * j / (num_intermediate_points + 1))
                    new_path_x.append(intermediate_x)
                    new_path_y.append(intermediate_y)

            # Add the current point to the new path
            new_path_x.append(path_x[i])
            new_path_y.append(path_y[i])
            print(f"new_path_x: {new_path_x}, new_path_y: {new_path_y}")
        return new_path_x, new_path_y

    def plan_path_based_mode(self, start_x, start_y, goal_x, goal_y):
        """
        Plan the path and return the x, y coordinates for the path.
        Merge points with very small incremental changes into a single point to reduce noise.
        Ensure the movement is constrained to up, down, left, or right (no diagonal moves).
        """
        threshold_distance = 0.1  # Threshold to merge small incremental changes
        min_distance = 0.2  # Minimum distance to avoid noise
        max_distance = 0.2  # Maximum distance before adding intermediate points

        # Get the planned path (x and y coordinates)
        path_x, path_y = self.plan_path(start_x, start_y, goal_x, goal_y)

        # Lists to store the new path with merged points
        merged_path_x, merged_path_y = [path_x[0]], [path_y[0]]  # Start with the first point

        for i in range(1, len(path_x)):
            # Calculate the distance between consecutive points
            dx = path_x[i] - path_x[i - 1]
            dy = path_y[i] - path_y[i - 1]

            # Ensure that the movement is only horizontal or vertical, no diagonals
            if dx != 0 and dy != 0:
                # If both dx and dy are non-zero, it means the movement is diagonal.
                # In this case, we'll break it into separate horizontal and vertical moves.
                if abs(dx) > abs(dy):
                    # Move horizontally first
                    intermediate_x = path_x[i - 1] + dx
                    merged_path_x.append(intermediate_x)
                    merged_path_y.append(path_y[i - 1])
                else:
                    # Move vertically first
                    intermediate_y = path_y[i - 1] + dy
                    merged_path_x.append(path_x[i - 1])
                    merged_path_y.append(intermediate_y)
                continue  # Move on to the next point after splitting

            # Calculate the distance for the current move
            distance = math.hypot(dx, dy)

            # Merge small incremental changes
            if distance < threshold_distance:
                # If it's the last point, we add it
                if i == len(path_x) - 1:
                    merged_path_x.append(path_x[i])
                    merged_path_y.append(path_y[i])
                continue  # Skip this point, we'll merge it with the next

            # If the distance exceeds the max threshold, add intermediate points
            if distance > max_distance:
                # Calculate the number of intermediate points to add
                num_intermediate_points = int(distance / max_distance)
                for j in range(1, num_intermediate_points + 1):
                    # Interpolate intermediate points (either horizontal or vertical)
                    if dx != 0:  # Horizontal move
                        intermediate_x = path_x[i - 1] + (dx * j / (num_intermediate_points + 1))
                        merged_path_x.append(intermediate_x)
                        merged_path_y.append(path_y[i - 1])
                    else:  # Vertical move
                        intermediate_y = path_y[i - 1] + (dy * j / (num_intermediate_points + 1))
                        merged_path_x.append(path_x[i - 1])
                        merged_path_y.append(intermediate_y)

            # Add the current point to the new path if it's not too small
            merged_path_x.append(path_x[i])
            merged_path_y.append(path_y[i])
            
        print(f"merged_path_x: {merged_path_x[::-1]}, \nmerged_path_y: {merged_path_y[::-1]}")
        return merged_path_x, merged_path_y




