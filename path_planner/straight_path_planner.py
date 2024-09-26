import math
import numpy as np
import matplotlib.pyplot as plt
import time
from path_planner.path_finding import PathPlanner       #when using by others file : from path_planner.path_finding import PathPlanner 
#from path_finding import PathPlanner 

class StraightPathPlanner(PathPlanner):
    def __init__(self, grid_resolution, robot_radius, target_radius, obstacle_size=0.1):
        # Call the parent constructor
        super().__init__(grid_resolution, robot_radius, target_radius, obstacle_size)
        self.motion_model = self.define_motion_model()

    def define_motion_model(self):
        """Override the motion model to restrict to straight-line movement."""
        step_size = 0.4  # Define grid step size for straight-line movement
        return [[step_size, 0, 1], [0, step_size, 1], [-step_size, 0, 1], [0, -step_size, 1]]  # Only horizontal/vertical moves

    def plan_path_based_mode(self, start_x, start_y, goal_x, goal_y):
        """Override the path planning method to ensure snapping to grid and straight-line movement."""
        # Snap start and goal positions to grid points
        start_x = self.snap_to_grid(start_x)
        start_y = self.snap_to_grid(start_y)
        goal_x = self.snap_to_grid(goal_x)
        goal_y = self.snap_to_grid(goal_y)

        # Use the parent class's plan_path method
        return self.plan_path(start_x, start_y, goal_x, goal_y)

    def snap_to_grid(self, position):
        """Snap a given position to the nearest grid point."""
        return np.round(position / self.grid_resolution) * self.grid_resolution
    
    @staticmethod
    def heuristic(node1, node2):
        """Override heuristic to use Manhattan distance for grid-based movement."""
        return abs(node1.x - node2.x) + abs(node1.y - node2.y)

    # @override
    def plan_path(self, start_x, start_y, goal_x, goal_y):
        """
        A* pathfinding algorithm for straight motion

        start_x, start_y: Starting coordinates
        goal_x, goal_y: Goal coordinates

        Returns the x and y positions of the final path.
        """
        start_node = self.Node(self.get_xy_index(start_x, self.grid_min_x),
                               self.get_xy_index(start_y, self.grid_min_y), 0.0, -1)
        goal_node = self.Node(self.get_xy_index(goal_x, self.grid_min_x),
                              self.get_xy_index(goal_y, self.grid_min_y), 0.0, -1)

        open_set, closed_set = dict(), dict()
        open_set[self.get_node_index(start_node)] = start_node

        while True:
            if len(open_set) == 0:
                raise Exception("No path found")

            current_id = min(open_set, key=lambda o: open_set[o].cost + self.heuristic(goal_node, open_set[o]))
            current_node = open_set[current_id]

            if math.hypot(current_node.x - goal_node.x, current_node.y - goal_node.y) <= self.target_radius + self.obstacle_size / 2:
                goal_node.parent_idx = current_node.parent_idx
                goal_node.cost = current_node.cost
                break

            del open_set[current_id]
            closed_set[current_id] = current_node

            for i, _ in enumerate(self.motion_model):
                new_node = self.Node(current_node.x + self.motion_model[i][0],
                                     current_node.y + self.motion_model[i][1],
                                     current_node.cost + self.motion_model[i][2], current_id)
                node_id = self.get_node_index(new_node)

                if not self.is_node_valid(new_node):
                    continue

                if node_id in closed_set:
                    continue

                if node_id not in open_set:
                    open_set[node_id] = new_node
                else:
                    if open_set[node_id].cost > new_node.cost:
                        open_set[node_id] = new_node

        path_x, path_y = self.extract_final_path(goal_node, closed_set)  # Extract the full path
        downsampled_x, downsampled_y = self.downsample_path(path_x, path_y)  # Correctly calling the downsample function
        return downsampled_x[1:], downsampled_y[1:]
    
    def downsample_path(self, path_x, path_y, min_dist=0.2):
        """
        Downsamples the path to reduce the number of points.
        min_dist: Minimum distance between consecutive points.
        """
        downsampled_x, downsampled_y = [path_x[0]], [path_y[0]]  # Start with the first point

        for i in range(1, len(path_x)):
            dist = math.hypot(path_x[i] - downsampled_x[-1], path_y[i] - downsampled_y[-1])
            if dist >= min_dist:
                downsampled_x.append(path_x[i])
                downsampled_y.append(path_y[i])

        # Ensure the goal point is always included
        if (path_x[-1], path_y[-1]) != (downsampled_x[-1], downsampled_y[-1]):
            downsampled_x.append(path_x[-1])
            downsampled_y.append(path_y[-1])

        return downsampled_x, downsampled_y