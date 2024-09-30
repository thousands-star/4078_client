import math
import numpy as np
import matplotlib.pyplot as plt
import time
from abc import ABC, abstractmethod
import json

class PathPlanner:

    def __init__(self, grid_resolution, robot_radius, target_radius, obstacle_size=0.1):
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
        self.grid_min_x, self.grid_min_y = -1.6, 1.6
        self.grid_max_x, self.grid_max_y = -1.6, 1.6
        self.obstacle_grid = None
        self.grid_width_x, self.grid_width_y = 0.05, 0.05  # Grid resolution in x and y
        self.obstacle_size = obstacle_size
        self.obstacle_x = []
        self.obstacle_y = []
        self.create_walls(-1.6, -1.6, 1.6, 1.6, 0.01)

    def set_mode(self, mode):
        self.mode = mode
    
    def get_mode(self):
        return self.mode
    
    def add_square_obstacle(self, x, y, size=0.08, resolution=0.01):
        """
        Add a square obstacle around the point (x, y).

        x, y: Center of the square
        size: Side length of the square
        resolution: Grid resolution for obstacle boundary points (0.08/0.01 = 8 points for boundary)
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
    @abstractmethod
    def plan_path(self, start_x, start_y, goal_x, goal_y):
        pass

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

    def get_grid_position(self, index, min_position):
        return index * self.grid_resolution + min_position

    def get_xy_index(self, position, min_pos):
        return round((position - min_pos) / self.grid_resolution)

    def get_node_index(self, node):
        return (node.y - self.grid_min_y) * self.grid_width_x + (node.x - self.grid_min_x)

    def is_node_valid(self, node):
        pos_x = self.get_grid_position(node.x, self.grid_min_x)
        pos_y = self.get_grid_position(node.y, self.grid_min_y)

        # Ensure node indices are integers
        node_x_index = int(node.x)
        node_y_index = int(node.y)

        # Check if the node is within the grid bounds
        if pos_x < self.grid_min_x or pos_y < self.grid_min_y or pos_x >= self.grid_max_x or pos_y >= self.grid_max_y:
            return False

        # Check if the node is within an expanded obstacle
        if self.obstacle_grid[node_x_index][node_y_index]:
            return False

        return True


    
    def add_obstacle(self, x, y, size):
        st = time.time()
        self.add_square_obstacle(x, y, size=size)
        print("time to add obstacles " + str(time.time()-st))
        st = time.time()
        self.build_obstacle_grid(self.obstacle_x, self.obstacle_y)
        print("time to build grid " + str(time.time()-st))

    def save_obstacle_grid(self, filename="obstacle_grid.json"):
        """Saves the obstacle grid and positions to a JSON file."""
        data = {
            "obstacle_x": self.obstacle_x,
            "obstacle_y": self.obstacle_y,
            "grid_min_x": self.grid_min_x,
            "grid_min_y": self.grid_min_y,
            "grid_max_x": self.grid_max_x,
            "grid_max_y": self.grid_max_y,
            "grid_width_x": self.grid_width_x,
            "grid_width_y": self.grid_width_y,
            "obstacle_grid": self.obstacle_grid
        }
        with open(filename, "w") as f:
            json.dump(data, f)
        print(f"Obstacle grid saved to {filename}")

    def load_obstacle_grid(self, filename="obstacle_grid.json"):
        """Loads the obstacle grid and positions from a JSON file."""
        with open(filename, "r") as f:
            data = json.load(f)
            self.obstacle_x = data["obstacle_x"]
            self.obstacle_y = data["obstacle_y"]
            self.grid_min_x = data["grid_min_x"]
            self.grid_min_y = data["grid_min_y"]
            self.grid_max_x = data["grid_max_x"]
            self.grid_max_y = data["grid_max_y"]
            self.grid_width_x = data["grid_width_x"]
            self.grid_width_y = data["grid_width_y"]
            self.obstacle_grid = data["obstacle_grid"]
        print(f"Obstacle grid loaded from {filename}")

    def add_obstacles(self, fruit_positions, aruco_positions, obstacle_size=0.1, grid_file="obstacle_grid.json"):
        """
        Adds obstacles to the planner's obstacle list from fruit positions and ArUco marker positions.
        If an obstacle grid is saved, it loads the grid from the file. Otherwise, it generates the grid.
        
        fruit_positions: List of fruit obstacle positions
        aruco_positions: List of ArUco marker positions
        obstacle_size: Size of each obstacle (square)
        """
        # try:
        #     # Try to load the obstacle grid from file
        #     self.load_obstacle_grid(grid_file)
        #     print("Loaded obstacle grid from file.")
        # except FileNotFoundError:
        #     # If the file is not found, generate the obstacle grid and save it
        #     print("Obstacle grid file not found, generating obstacles...")
        #     # Add fruit obstacles
        #     for fruit in fruit_positions:
        #         self.add_obstacle(fruit[0], fruit[1], obstacle_size)

        #     # Add ArUco marker obstacles
        #     for ox, oy in zip(aruco_positions[:, 0], aruco_positions[:, 1]):
        #         self.add_obstacle(ox, oy, obstacle_size)

        #     # Save the generated obstacle grid to file
        #     self.save_obstacle_grid(grid_file)
        # Add fruit obstacles
        for fruit in fruit_positions:
            self.add_obstacle(fruit[0], fruit[1], obstacle_size)

        # Add ArUco marker obstacles
        for ox, oy in zip(aruco_positions[:, 0], aruco_positions[:, 1]):
            self.add_obstacle(ox, oy, obstacle_size)



    def update_robot_radius(self, robot_radius):
        self.robot_radius = robot_radius
        self.build_obstacle_grid(self.obstacle_x, self.obstacle_y)

    def update_goal_radius(self, goal_radius):
        self.target_radius = goal_radius
        self.build_obstacle_grid(self.obstacle_x, self.obstacle_y)

    def build_obstacle_grid(self, ox, oy):
        # Create the obstacle grid using NumPy arrays
        self.grid_min_x = round(min(ox))
        self.grid_min_y = round(min(oy))
        self.grid_max_x = round(max(ox))
        self.grid_max_y = round(max(oy))

        # Define grid dimensions
        self.grid_width_x = int((self.grid_max_x - self.grid_min_x) / self.grid_resolution)
        self.grid_width_y = int((self.grid_max_y - self.grid_min_y) / self.grid_resolution)

        # Initialize obstacle grid with False
        self.obstacle_grid = np.zeros((self.grid_width_x, self.grid_width_y), dtype=bool)

        # Get the range of cells affected by each obstacle
        for obs_x, obs_y in zip(ox, oy):
            obs_ix = self.get_xy_index(obs_x, self.grid_min_x)
            obs_iy = self.get_xy_index(obs_y, self.grid_min_y)

            # Calculate the radius in grid cells, considering both robot radius and obstacle size
            radius = int((self.robot_radius + self.obstacle_size / 2) / self.grid_resolution)

            # Define the range of cells to be affected by this obstacle
            min_ix = max(0, obs_ix - radius)
            max_ix = min(self.grid_width_x - 1, obs_ix + radius)
            min_iy = max(0, obs_iy - radius)
            max_iy = min(self.grid_width_y - 1, obs_iy + radius)

            # Mark the cells within the radius as obstacles
            for ix in range(min_ix, max_ix + 1):
                for iy in range(min_iy, max_iy + 1):
                    if math.hypot(self.get_grid_position(ix, self.grid_min_x) - obs_x,
                                self.get_grid_position(iy, self.grid_min_y) - obs_y) <= self.robot_radius + (self.obstacle_size / 2):
                        self.obstacle_grid[ix][iy] = True


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

    def reset_obstacles(self):
        self.obstacle_x = []
        self.obstacle_y = []
        self.create_walls(-1.6, -1.6, 1.6, 1.6, 0.01)
        
    @staticmethod
    def compute_goal_positions(fruit_list, fruit_positions, search_list):
        goals = []
        for fruit in search_list:
            if fruit in fruit_list:
                # If the fruit is in the fruit list, add its position to the goals
                fruit_index = fruit_list.index(fruit)
                goals.append(fruit_positions[fruit_index])
            else:
                # If the fruit is not in the fruit list, handle it as needed
                print(f"Warning: {fruit} not found in the fruit list. It will need to be detected dynamically.")
                # Optionally, you can append None or handle this fruit dynamically
                # goals.append(None)  # Or leave it out to handle dynamically later
        return goals


    ###################################################################
    @abstractmethod
    def heuristic(node1, node2):
        pass

    @abstractmethod
    def define_motion_model(self):
        """Abstract method for defining the motion model. Must be implemented by the subclass."""
        pass

    @abstractmethod
    def plan_path_based_mode(self, start_x, start_y, goal_x, goal_y):
        pass
    
    def replan(self, new_x, new_y, goal_x, goal_y):
        return self.plan_path_based_mode(new_x, new_y, goal_x, goal_y)

    def are_points_collinear(self, x1, y1, x2, y2, x3, y3):
        """
        Check if three points are collinear (i.e., they lie on a straight line).
        """
        # Using the slope formula to check if the points are aligned
        # (y2 - y1) / (x2 - x1) should be equal to (y3 - y2) / (x3 - x2)
        # This avoids division by zero, and we just cross-multiply to check for collinearity
        return abs((y2 - y1) * (x3 - x2) - (y3 - y2) * (x2 - x1)) < 1e-6
        
    def plot_obstacle_grid(self):
        """
        Plot the obstacle grid for visualization purposes without the black part.
        Only the obstacles are displayed using scatter.
        """
        if self.obstacle_grid is None:
            print("No obstacle grid to plot.")
            return

        # Create a figure and axis for plotting
        plt.figure(figsize=(8, 8))

        # Overlay the obstacle points using scatter
        plt.scatter(self.obstacle_x, self.obstacle_y, color='red', s=10, label="Obstacles")  # s controls the size of the scatter points

        # Add grid lines
        plt.grid(True)
        plt.title("Obstacle Grid")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.axis("equal")
        plt.gca().set_xticks(np.arange(-1.6, 1.6, 0.4))
        plt.gca().set_yticks(np.arange(-1.6, 1.6, 0.4))
        plt.xlim(-1.6, 1.6)
        plt.ylim(-1.6, 1.6)

        # Show the plot
        plt.legend()
        plt.show()

    def update_obstacle(self, fruit_type, new_position, size=None):
        """
        Updates the position of an obstacle in the path planner.
        
        Parameters:
        - fruit_type: The type of fruit (used as an identifier for the obstacle).
        - new_position: The new position of the obstacle as [x, y].
        - size: Optional size of the obstacle. If not provided, the default obstacle size will be used.
        """
        # Default to the obstacle size if not specified
        size = size if size else self.obstacle_size

        # Check if the fruit's previous position exists in the obstacle list
        for i in range(len(self.obstacle_x)):
            # You can use a threshold to match the position (due to float precision errors)
            if math.isclose(self.obstacle_x[i], new_position[0], abs_tol=0.01) and math.isclose(self.obstacle_y[i], new_position[1], abs_tol=0.01):
                print(f"Removing old obstacle at ({self.obstacle_x[i]}, {self.obstacle_y[i]}) for {fruit_type}")
                # Remove the old obstacle
                self.obstacle_x.pop(i)
                self.obstacle_y.pop(i)
                break

        # Add the new obstacle position
        print(f"Adding new obstacle at {new_position} for {fruit_type}")
        self.add_square_obstacle(new_position[0], new_position[1], size=size)

        # Rebuild the obstacle grid to reflect the updated positions
        self.build_obstacle_grid(self.obstacle_x, self.obstacle_y)

class MathTools():
    @staticmethod
    def find_turning_angle(waypoint, robot_pose):
        ########## TURN ##########        
        dx = waypoint[0] - robot_pose[0]
        dy = waypoint[1] - robot_pose[1]
        theta_r = robot_pose[2]                         # angle of robot from origin
        theta_w = np.arctan2(dy, dx)                    # angle of waypoint from robot
        theta_turn = theta_w - theta_r                  # angle to be turned by robot
        theta_turn = (theta_turn + np.pi) % (2 * np.pi) - np.pi  # normalize angle to [-pi, pi], make sure it always turns the smallest angle
        theta_deg = float(theta_turn * 180 / np.pi)
        print(f"Angle calculated: {theta_w} - {theta_r} = {theta_turn}")
        return theta_deg
    
    @staticmethod
    def find_euclidean_distance(waypoint, robot_pose):
        ########## GO STRAIGHT ##########
        # Parameters to go straight
        # wheel_speed_strg = 0.7
        # Calculate the distance to move
        dx = waypoint[0] - robot_pose[0]
        dy = waypoint[1] - robot_pose[1]
        d_move = np.sqrt(dx**2 + dy**2)
        return d_move
    
    @staticmethod
    def rad(deg):
        return deg/180 * np.pi
    
    @staticmethod
    def deg(rad):
        return rad / np.pi * 180
    
    @staticmethod
    #constrain the angle in pi to -pi
    def clamp_angle(rad):
        # Clamp the angle between -π and π
        while rad > np.pi:
            rad -= 2 * np.pi
        while rad < -np.pi:
            rad += 2 * np.pi
        return rad
    
class MapReader:
    def __init__(self, map_fname = None, search_list = None) -> None:
        self.map_fname = map_fname
        self.search_list = search_list
        pass

    def load_map_data(self):
        filename = self.map_fname
        with open(filename, 'r') as file:
            data = json.load(file)
            fruits, fruit_positions, aruco_positions = [], [], np.empty([10, 2])

            for key, val in data.items():
                x = np.round(val['x'], 2)
                y = np.round(val['y'], 2)

                if key.startswith('aruco'):
                    marker_id = 9 if key.startswith('aruco10') else int(key[5]) - 1
                    aruco_positions[marker_id] = [x, y]
                else:
                    fruits.append(key[:-2])
                    if len(fruit_positions) == 0:
                        fruit_positions = np.array([[x, y]])
                    else:
                        fruit_positions = np.append(fruit_positions, [[x, y]], axis=0)

            return fruits, fruit_positions, aruco_positions
        
    def read_search_list(self):
        """
        Read the search order of the target fruits
        @return: search order of the target fruits
        """
        search_list = []
        with open(self.search_list, 'r') as fd:
            fruits = fd.readlines()

            for fruit in fruits:
                search_list.append(fruit.strip())

        return search_list
    
    def compute_goal_positions(self, fruit_list, fruit_positions, search_list):
        return [fruit_positions[fruit_list.index(fruit)] for fruit in search_list]
    

   