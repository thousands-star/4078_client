import math
import numpy as np
import matplotlib.pyplot as plt
import time
try:
    from path_planner.path_finding import PathPlanner
except ImportError:
    from path_finding import PathPlanner 

class DiagonalPathPlanner(PathPlanner):
    def __init__(self, grid_resolution, robot_radius, target_radius, obstacle_size=0.1):
        super().__init__(grid_resolution, robot_radius, target_radius, obstacle_size)
        self.motion_model = self.define_motion_model()

    @staticmethod
    def heuristic(node1, node2):
        "Manhattan distance"
        return math.hypot(node1.x - node2.x, node1.y - node2.y)
    
    def define_motion_model(self):
        return [[1, 0, 1], [0, 1, 1], [-1, 0, 1], [0, -1, 1],
                [-1, -1, math.sqrt(2)], [-1, 1, math.sqrt(2)],
                [1, -1, math.sqrt(2)], [1, 1, math.sqrt(2)]]
    
    def plan_path(self, start_x, start_y, goal_x, goal_y):
        """
        A* pathfinding algorithm for diagonal motion

        start_x, start_y: Starting coordinates
        goal_x, goal_y: Goal coordinates

        Returns the x and y positions of the final path.
        """
        startTime = time.time()
        # Debug statement
        # print(self.get_xy_index(start_x, self.grid_min_x))
        start_node = self.Node(self.get_xy_index(start_x, self.grid_min_x),
                               self.get_xy_index(start_y, self.grid_min_y), 0.0, -1)
        goal_node = self.Node(self.get_xy_index(goal_x, self.grid_min_x),
                              self.get_xy_index(goal_y, self.grid_min_y), 0.0, -1)
        
        open_set, closed_set = dict(), dict()
        open_set[self.get_node_index(start_node)] = start_node

        lv = 0
        while True:

            if (len(open_set) == 0):
                print("Open set is empty.")
                raise Exception("No path found")

            current_id = min(open_set, key=lambda o: open_set[o].cost + self.heuristic(goal_node, open_set[o]))
            current_node = open_set[current_id]
            show_animation = False
            if show_animation:  # pragma: no cover
                plt.plot(self.get_grid_position(current_node.x, self.grid_min_x),
                        self.get_grid_position(current_node.y, self.grid_min_y), "xc")
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect('key_release_event',
                                            lambda event: [exit(
                                                0) if event.key == 'escape' else None])
                if len(closed_set.keys()) % 10 == 0:
                    plt.pause(0.01)

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

        path_x, path_y = self.extract_final_path(goal_node, closed_set)
        endTime = time.time()
        print("Navigation Time: " + str(endTime-startTime))
        return path_x[1:], path_y[1:]
    
    def plan_path_based_mode(self, start_x, start_y, goal_x, goal_y):
        return self.generate_smooth_path(start_x, start_y, goal_x, goal_y)
    

    ### Helper functions to smooth the path
    def find_turning_points(self, path_x, path_y, threshold_angle=10):
        """
        Find turning points based on the angle between three consecutive points.
        
        threshold_angle: Angle in degrees, points with an angle difference greater than this value will be counted as turning points.
        """
        if len(path_x) < 3:
            # If the path has fewer than 3 points, no turning points can be found
            return path_x, path_y

        turning_points_x = [path_x[0]]  # Start point
        turning_points_y = [path_y[0]]

        def angle_between(p1, p2, p3):
            """Calculate the angle between three points (p1, p2, p3)."""
            v1 = (p1[0] - p2[0], p1[1] - p2[1])  # Vector 1: p1 -> p2
            v2 = (p3[0] - p2[0], p3[1] - p2[1])  # Vector 2: p2 -> p3
            dot_product = v1[0] * v2[0] + v1[1] * v2[1]
            mag_v1 = math.hypot(v1[0], v1[1])
            mag_v2 = math.hypot(v2[0], v2[1])
            if mag_v1 == 0 or mag_v2 == 0:
                return 0
            cos_theta = dot_product / (mag_v1 * mag_v2)
            # Clamp cos_theta to the range [-1, 1] to avoid math domain errors
            cos_theta = max(-1, min(1, cos_theta))
            
            return math.acos(cos_theta) * 180 / math.pi

        for i in range(1, len(path_x) - 1):
            x1, y1 = path_x[i - 1], path_y[i - 1]
            x2, y2 = path_x[i], path_y[i]
            x3, y3 = path_x[i + 1], path_y[i + 1]

            # Calculate angle between points
            angle_diff = angle_between((x1, y1), (x2, y2), (x3, y3))

            # If the angle difference is greater than the threshold, it's a turning point
            if angle_diff > threshold_angle:
                turning_points_x.append(x2)
                turning_points_y.append(y2)

        # Append final point
        turning_points_x.append(path_x[-1])
        turning_points_y.append(path_y[-1])

        return turning_points_x, turning_points_y

    def smooth_path(self, path_x, path_y):
        """
        Path smoothing function to reduce waypoints by checking line-of-sight between points.
        
        It removes intermediate points if the path between two points is free of obstacles.
        """
        if len(path_x) < 2:
            return path_x, path_y

        smoothed_path_x = [path_x[0]]
        smoothed_path_y = [path_y[0]]
        
        i = 0
        while i < len(path_x) - 1:
            j = len(path_x) - 1
            found = False
            while j > i:
                if self.check_collision(smoothed_path_x[-1], smoothed_path_y[-1], path_x[j], path_y[j]):
                    j -= 1
                else:
                    smoothed_path_x.append(path_x[j])
                    smoothed_path_y.append(path_y[j])
                    i = j
                    found = True
                    break
            if not found:
                # Prevent an infinite loop if no valid j is found
                i += 1

        # Ensure the final point is included
        smoothed_path_x.append(path_x[-1])
        smoothed_path_y.append(path_y[-1])

        # print("Smoothed Path X, Y:")
        # for x, y in zip(smoothed_path_x[::-1], smoothed_path_y[::-1]):
        #     print(f"({x:.2f}, {y:.2f})")  # Print coordinates with 2 decimal places
            
        return smoothed_path_x, smoothed_path_y
    
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
        Bresenham's line algorithm to find grid points between two coordinates.
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

    def generate_smooth_path(self, start_x, start_y, goal_x, goal_y, threshold_angle=15):
        """
        Generates a smoothed path from start to goal by planning the path, 
        finding turning points, and smoothing the path.

        start_x, start_y: Starting coordinates
        goal_x, goal_y: Goal coordinates
        threshold_angle: Angle threshold to define turning points
        """
        # Plan the path using A* algorithm
        path_x, path_y = self.plan_path(start_x, start_y, goal_x, goal_y)

        # Find turning points in the path
        turning_x, turning_y = self.find_turning_points(path_x, path_y, threshold_angle)

        # Smooth the path by removing unnecessary waypoints
        smoothed_x, smoothed_y = self.smooth_path(turning_x, turning_y)

        return smoothed_x, smoothed_y
    