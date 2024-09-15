import math
import numpy as np
import matplotlib.pyplot as plt
import time



class PathPlanner:

    def __init__(self, grid_resolution, robot_radius, target_radius=0.1, obstacle_size=0.1):
        """
        Initialize grid for A* path planning

        obs_x: List of x coordinates of obstacles
        obs_y: List of y coordinates of obstacles
        grid_resolution: Grid resolution in meters
        robot_radius: Robot radius in meters
        """
        self.target_radius = target_radius  # Threshold radius around the goal
        self.grid_resolution = grid_resolution  # Grid resolution
        self.robot_radius = robot_radius  # Robot's safety radius
        # Defining the grid bounds
        self.grid_min_x, self.grid_min_y = -1.5, 1.5
        self.grid_max_x, self.grid_max_y = -1.5, 1.5
        self.obstacle_grid = None
        self.grid_width_x, self.grid_width_y = 0.05, 0.05  # Grid resolution in x and y
        self.motion_model = self.define_motion_model()
        self.obstacle_size = obstacle_size

        self.obstacle_x = []
        self.obstacle_y = []
        self.create_walls(-1.5, -1.5, 1.5, 1.5, 0.01)


    def add_square_obstacle(self, x, y, size=0.08, resolution=0.01):
        """
        Add a square obstacle around the point (x, y).

        x, y: Center of the square
        size: Side length of the square
        resolution: Grid resolution for obstacle boundary points
        """
        half_size = size / 2

        top_left = (x - half_size, y - half_size)
        top_right = (x + half_size, y - half_size)
        bottom_left = (x - half_size, y + half_size)
        bottom_right = (x + half_size, y + half_size)

        for i in np.arange(top_left[0], top_right[0], resolution):
            self.obstacle_x.append(i)
            self.obstacle_y.append(top_left[1])

        for i in np.arange(bottom_left[0], bottom_right[0], resolution):
            self.obstacle_x.append(i)
            self.obstacle_y.append(bottom_left[1])

        for i in np.arange(top_left[1], bottom_left[1], resolution):
            self.obstacle_x.append(top_left[0])
            self.obstacle_y.append(i)

        for i in np.arange(top_right[1], bottom_right[1], resolution):
            self.obstacle_x.append(top_right[0])
            self.obstacle_y.append(i)

    class Node:
        def __init__(self, x, y, cost, parent_idx):
            self.x = x  # x index in the grid
            self.y = y  # y index in the grid
            self.cost = cost
            self.parent_idx = parent_idx

        def __str__(self):
            return f"{self.x},{self.y},{self.cost},{self.parent_idx}"

    def plan_path(self, start_x, start_y, goal_x, goal_y):
        """
        A* pathfinding algorithm

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

            if math.hypot(current_node.x - goal_node.x, current_node.y - goal_node.y) <= self.target_radius:
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

    def extract_final_path(self, goal_node, closed_set):
        path_x, path_y = [self.get_grid_position(goal_node.x, self.grid_min_x)], [
            self.get_grid_position(goal_node.y, self.grid_min_y)]
        parent_idx = goal_node.parent_idx
        while parent_idx != -1:
            node = closed_set[parent_idx]
            path_x.append(self.get_grid_position(node.x, self.grid_min_x))
            path_y.append(self.get_grid_position(node.y, self.grid_min_y))
            parent_idx = node.parent_idx

        return path_x, path_y

    @staticmethod
    def heuristic(node1, node2):
        return math.hypot(node1.x - node2.x, node1.y - node2.y)

    def get_grid_position(self, index, min_position):
        return index * self.grid_resolution + min_position

    def get_xy_index(self, position, min_pos):
        return round((position - min_pos) / self.grid_resolution)

    def get_node_index(self, node):
        return (node.y - self.grid_min_y) * self.grid_width_x + (node.x - self.grid_min_x)

    def is_node_valid(self, node):
        pos_x = self.get_grid_position(node.x, self.grid_min_x)
        pos_y = self.get_grid_position(node.y, self.grid_min_y)

        if pos_x < self.grid_min_x or pos_y < self.grid_min_y or pos_x >= self.grid_max_x or pos_y >= self.grid_max_y:
            return False

        if self.obstacle_grid[node.x][node.y]:
            return False

        return True

    def add_obstacle(self, x, y, size):
        st = time.time()
        self.add_square_obstacle(x, y, size=size)
        print("time to add obstacles " + str(time.time()-st))
        st = time.time()
        self.build_obstacle_grid(self.obstacle_x, self.obstacle_y)
        print("time to build grid " + str(time.time()-st))

    def update_robot_radius(self, robot_radius):
        self.robot_radius = robot_radius
        self.build_obstacle_grid(self.obstacle_x, self.obstacle_y)

    def update_goal_radius(self, goal_radius):
        self.target_radius = goal_radius
        self.build_obstacle_grid(self.obstacle_x, self.obstacle_y)

    def build_obstacle_grid(self, ox, oy):
        self.grid_min_x = round(min(ox))
        self.grid_min_y = round(min(oy))
        self.grid_max_x = round(max(ox))
        self.grid_max_y = round(max(oy))

        self.grid_width_x = round((self.grid_max_x - self.grid_min_x) / self.grid_resolution)
        self.grid_width_y = round((self.grid_max_y - self.grid_min_y) / self.grid_resolution)

        self.obstacle_grid = [[False for _ in range(self.grid_width_y)]
                              for _ in range(self.grid_width_x)]
        for ix in range(self.grid_width_x):
            x = self.get_grid_position(ix, self.grid_min_x)
            for iy in range(self.grid_width_y):
                y = self.get_grid_position(iy, self.grid_min_y)
                for obs_x, obs_y in zip(ox, oy):
                    if math.hypot(obs_x - x, obs_y - y) <= self.robot_radius:
                        self.obstacle_grid[ix][iy] = True
                        break

    def create_walls(self, min_x, min_y, max_x, max_y, resolution):
        for i in np.arange(min_x, max_x, resolution):
            self.obstacle_x.append(i)
            self.obstacle_y.append(min_y)
        for i in np.arange(min_y, max_y, resolution):
            self.obstacle_x.append(max_x)
            self.obstacle_y.append(i)
        for i in np.arange(min_x, max_x + resolution, resolution):
            self.obstacle_x.append(i)
            self.obstacle_y.append(max_y)
        for i in np.arange(min_y, max_y + resolution, resolution):
            self.obstacle_x.append(min_x)
            self.obstacle_y.append(i)

    @staticmethod
    def define_motion_model():
        return [[1, 0, 1], [0, 1, 1], [-1, 0, 1], [0, -1, 1],
                [-1, -1, math.sqrt(2)], [-1, 1, math.sqrt(2)],
                [1, -1, math.sqrt(2)], [1, 1, math.sqrt(2)]]

    def reset_obstacles(self):
        self.obstacle_x = []
        self.obstacle_y = []
        self.create_walls(-1.5, -1.5, 1.5, 1.5, 0.01)
        
    @staticmethod
    def compute_goal_positions(fruit_list, fruit_positions, search_list):
        return [fruit_positions[fruit_list.index(fruit)] for fruit in search_list]

    def find_turning_points(self, path_x, path_y, threshold_angle=10):
        
        turning_points_x = [path_x[0]]
        turning_points_y = [path_y[0]]

        for i in range(1, len(path_x) - 1):
            x1, y1 = path_x[i - 1], path_y[i - 1]
            x2, y2 = path_x[i], path_y[i]
            x3, y3 = path_x[i + 1], path_y[i + 1]

            angle1 = math.atan2(y2 - y1, x2 - x1)
            angle2 = math.atan2(y3 - y2, x3 - x2)

            angle_diff = abs(angle1 - angle2) * 180 / math.pi

            # Debugging prints
            print(f"Point {i}: ({x2}, {y2})")
            print(f"Angle1: {angle1}, Angle2: {angle2}, Angle Difference: {angle_diff}")

            if angle_diff > threshold_angle:
                turning_points_x.append(x2)
                turning_points_y.append(y2)

        turning_points_x.append(path_x[-1])
        turning_points_y.append(path_y[-1])

        return turning_points_x, turning_points_y
    
    def export_obstacle_grid(self, filename="obstacle_grid.txt"):
        with open(filename, 'w') as f:
            f.write(f"{self.grid_min_x},{self.grid_min_y},{self.grid_max_x},{self.grid_max_y}\n")
            f.write(f"{self.grid_width_x},{self.grid_width_y}\n")
            for row in self.obstacle_grid:
                f.write(','.join(map(str, row)) + '\n')
    
    # Besides importing obstacle_grid
    # also instantiate mapReader, and get fruit & fruit_location & aruco_location
    # use add_square_obstacles to add things inside
    def import_obstacle_grid(self, filename="obstacle_grid.txt"):
        with open(filename, 'r') as f:
            # Read grid bounds and dimensions
            self.grid_min_x, self.grid_min_y, self.grid_max_x, self.grid_max_y = map(int, f.readline().split(','))
            self.grid_width_x, self.grid_width_y = map(int, f.readline().split(','))
            
            # Initialize the obstacle grid
            self.obstacle_grid = []
            for line in f:
                row = list(map(lambda x: x == 'True', line.strip().split(',')))
                self.obstacle_grid.append(row)