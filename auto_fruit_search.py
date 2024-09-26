# M4 - Autonomous fruit searching
import os
import sys
import cv2
import ast
import json
import time
import argparse
import numpy as np
from pibot import PibotControl
from pibot import Drive
import matplotlib.pyplot as plt
import math

# import SLAM components (M2)
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF 
from slam.robot import Robot
import slam.aruco_sensor as aruco

# import CV components (M3)
sys.path.insert(0,"{}/cv/".format(os.getcwd()))
from cv.detector import *
from cv.predictor import Fruit_Predictor

# import Path Planning components (M4)
from path_planner.path_finding import MapReader
from path_planner.path_finding import PathPlanner
from path_planner.diagonal_path_planner import DiagonalPathPlanner
from path_planner.straight_path_planner import StraightPathPlanner
from path_planner.path_finding import MathTools as mt

class Command_Controller:
    def __init__(self, 
                 ekf:EKF, 
                 aruco:aruco.ArucoSensor, 
                 control:PibotControl, 
                 detector:ObjectDetector,
                 predictor:Fruit_Predictor
                 ):
        self.ekf = ekf
        self.aruco_sensor = aruco
        self.control = control
        self.img = None
        self.detector = detector
        self.predictor = predictor
        self.current_goal = 0
        # Load the map onto the ekf
        ekf.switch_off_updating()
        self.control.set_mode(1)

    # SLAM with ARUCO markers       
    def get_new_state(self, drive_meas, period = 20):
        self.img = self.control.get_image()
        measurements, self.aruco_img = self.aruco_sensor.detect_marker_positions(self.img)
        # Drop camera model when it detected less than 2 marker
        if(len(measurements) < 2):
            measurements = []
        # # Computer Vision
        # # Perform YOLO detection using the YOLODetector
        # formatted_output, _ = self.detector.detect_single_image(self.img)
        # for prediction in formatted_output:
        #     fruit_type, x_center, y_center, width, height = prediction
        #     position = self.get_fruit_position_relative_to_camera(prediction)
        #     print(f"Detected {fruit_type} at camera position {position}")

        self.ekf.predict(drive_meas)    # predict ekf
        self.ekf.add_landmarks(measurements)    # add landmarks detected to ekf
        self.ekf.update(measurements)   # update ekf based on landmarks (Correct the predict state)
        robot_state = self.ekf.get_state_vector()   # get the current state vector
        robot_state[2] = mt.clamp_angle(robot_state[2])
        print(f"Robot State: {robot_state[0]},{robot_state[1]},{mt.deg(robot_state[2])}")
    
        for i in range(period-1):
            self.img = self.control.get_image()
            measurements, self.aruco_img = self.aruco_sensor.detect_marker_positions(self.img)
            # Drop camera model when it detected less than 2 marker
            if(len(measurements) < 3):
                measurements = []
            self.ekf.predict(Drive(0,0,0))
            self.ekf.add_landmarks(measurements)
            self.ekf.update(measurements)
            time.sleep(0.01)
            # robot_state = self.ekf.get_state_vector()
            # robot_state[2] = mt.clamp_angle(robot_state[2])
            # print(f"Robot State: {robot_state[0]},{robot_state[1]},{mt.deg(robot_state[2])}")

        robot_state = self.ekf.get_state_vector()
        robot_state[2] = mt.clamp_angle(robot_state[2])
        print(f"After {period} steps EKF: ")
        print(f"Robot State: {robot_state[0]},{robot_state[1]},{mt.deg(robot_state[2])}")
        return robot_state
        
    # A feedback mechanism with ekf and camera model and control it to each waypoint.
    def navigate(self, waypoint, is_turning_point=False):
        robot_state = self.ekf.get_state_vector()
        px = waypoint[0]
        py = waypoint[1]

        print(f"Waypoint: {px}, {py}")

        # Calculate the distance and angle to the next waypoint
        theta = mt.find_turning_angle([px, py], robot_state)
        dist = mt.find_euclidean_distance([px, py], robot_state)
        print(f"Distance: {dist}, Angle: {theta}")

        if is_turning_point:
            # Prioritize precise rotation at turning points
            print("\nAt a turning point, rotating precisely...")
            while abs(theta) > 3:  # Adjust this threshold for fine rotation control
                # Rotate to correct the angle
                self.control.set_angle_deg(theta)
                self.ekf.set_var = [0.15, 0.01]
                radius = 0.06
                disp = abs(mt.rad(theta) * radius)
                if theta > 0:
                    left_speed = -0.8   # based on the listen_2 turning speed
                    right_speed = 0.8
                else:
                    left_speed = 0.8
                    right_speed = -0.8
                drive_meas = Drive(left_speed, right_speed, disp / 0.8)
                robot_state = self.get_new_state(drive_meas=drive_meas)
                theta = mt.find_turning_angle([px, py], robot_state)
                print(f"Correcting angle at turning point: {theta}")

        # Continue moving towards the waypoint, without rotation unless it's a turning point
        if dist > 0.05:
            # Move forward
            left_speed = 0.6
            right_speed = 0.6
            self.ekf.set_var = [0.01, 0.1]
            self.control.set_displacement(dist[0])
            disp = dist
            drive_meas = Drive(left_speed, right_speed, disp / 0.6)
            robot_state = self.get_new_state(drive_meas=drive_meas)
            dist = mt.find_euclidean_distance([px, py], robot_state)


## Helper function
def plot_waypoints_for_goal(path_planner, waypoint_x, waypoint_y, search_target, show=True):
    """
    Plots the obstacles and the waypoints for a specific goal on the same grid.

    Parameters:
    - path_planner: The path planner object containing the obstacle grid.
    - waypoint_x: List of x-coordinates for the waypoints.
    - waypoint_y: List of y-coordinates for the waypoints.
    - search_target: The name of the search target.
    - show: Whether to show the plot after plotting.
    """
    # Plot the obstacles first
    if path_planner.obstacle_grid is None:
        print("No obstacle grid to plot.")
        return

    # Create a figure and axis for plotting (make sure to keep it persistent)
    plt.figure(figsize=(8, 8))

    # Overlay the obstacle points using scatter
    plt.scatter(path_planner.obstacle_x, path_planner.obstacle_y, color='red', s=10, label="Obstacles")  # s controls the size of the scatter points
    plt.plot(start_x, start_y, "og", markersize=10, label='Start Position')
    plt.plot(goal[0], goal[1], "xb")
    # Check if waypoints exist and are not empty
    if len(waypoint_x) == 0 or len(waypoint_y) == 0:
        print(f"No waypoints found for {search_target}.")
        return

    # Plot the waypoints for the current goal
    plt.plot(waypoint_x, waypoint_y, marker='o', linestyle='-', color='b', label=f"Path to {search_target}")

    # Highlight start and end points
    plt.scatter(waypoint_x[0], waypoint_y[0], color='green', label='Start')
    plt.scatter(waypoint_x[-1], waypoint_y[-1], color='red', label='Goal')

    # Add labels and a title to the plot
    plt.title(f"Path to {search_target}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)

    # Show the plot if required
    plt.show()

def find_turning_points(path_x, path_y, threshold_angle=15, min_distance=0.1):
    """
    Find turning points based on the angle between three consecutive points,
    excluding points that are collinear or have very small angles.
    
    path_x: List of x-coordinates of the waypoints.
    path_y: List of y-coordinates of the waypoints.
    threshold_angle: Angle threshold in degrees to determine turning points.
    min_distance: Minimum distance between waypoints to consider for turning.

    Returns:
        turning_points_x: List of x-coordinates of the turning points.
        turning_points_y: List of y-coordinates of the turning points.
    """
    if len(path_x) < 3:
        # If there are fewer than 3 points, no turning points can be found
        return [], []

    turning_points_x = [path_x[0]]  # Start with the first point
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
        return math.acos(cos_theta) * 180 / math.pi

    def are_points_collinear(p1, p2, p3, tolerance=1e-6):
        """Check if three points are collinear (i.e., they lie on the same straight line)."""
        # Using the slope formula to check if the points are aligned
        # (y2 - y1) / (x2 - x1) should be equal to (y3 - y2) / (x3 - x2)
        # This avoids division by zero, and we just cross-multiply to check for collinearity
        return abs((p2[1] - p1[1]) * (p3[0] - p2[0]) - (p3[1] - p2[1]) * (p2[0] - p1[0])) < tolerance

    # Iterate over the waypoints and find turning points
    for i in range(1, len(path_x) - 1):
        x1, y1 = path_x[i - 1], path_y[i - 1]
        x2, y2 = path_x[i], path_y[i]
        x3, y3 = path_x[i + 1], path_y[i + 1]

        # Check if the three points are collinear
        if are_points_collinear((x1, y1), (x2, y2), (x3, y3)):
            continue  # Skip if they are collinear

        # Calculate angle between points
        angle_diff = angle_between((x1, y1), (x2, y2), (x3, y3))

        # Only consider turning points if the angle is above the threshold
        dist1 = math.hypot(x2 - x1, y2 - y1)
        dist2 = math.hypot(x3 - x2, y3 - y2)

        if angle_diff > threshold_angle and dist1 > min_distance and dist2 > min_distance:
            turning_points_x.append(x2)
            turning_points_y.append(y2)

    # Append the final point
    turning_points_x.append(path_x[-1])
    turning_points_y.append(path_y[-1])

    return turning_points_x, turning_points_y


# main loop
if __name__ == "__main__":
    # argument parsing for command-line execution
    parser = argparse.ArgumentParser("Fruit searching")
    parser.add_argument("--map", type=str, default='truemap_cv.txt')
    parser.add_argument("--search_list", type=str, default='search_list.txt')
    parser.add_argument("--ip", metavar='', type=str, default='192.168.137.59')
    parser.add_argument("--port", metavar='', type=int, default=5000)
    parser.add_argument("--calib_dir", type=str, default="calibration/param/", help="Directory containing calibration parameters")
    parser.add_argument("--yolo_model", default='YOLO/model/yolov8_model.pt', help="YOLO model file for object detection")

    args, _ = parser.parse_known_args()

    # start timer
    run_timer = time.time()

    # M2 Initialization
    # Initialize the Pibot Control
    print("--------------------Initializing ROBOT & EKF--------------------")
    pictrl = PibotControl(args.ip,args.port)

    # Load camera and wheel calibration parameters
    camera_matrix = np.loadtxt(f"{args.calib_dir}/intrinsic.txt", delimiter=',')
    dist_coeffs = np.loadtxt(f"{args.calib_dir}/distCoeffs.txt", delimiter=',')
    scale = np.loadtxt(f"{args.calib_dir}/scale.txt", delimiter=',')
    baseline = np.loadtxt(f"{args.calib_dir}/baseline.txt", delimiter=',')
    
    # Initialize the Robot object
    robot = Robot(baseline, scale, camera_matrix, dist_coeffs)

    # Initialize EKF with the robot object and provided true map.
    ekf = EKF(robot, truemap="truemap.txt")

    # Initialize the ArUco detector (for updating pose based on markers)
    aruco_detector = aruco.ArucoSensor(robot, marker_length=0.06)

    print("====================Initializing Fruit Detector====================")
            # Initialise detector (M3)
    obj_detector = YOLODetector(r"cv\model\YOLO_best.pt", use_gpu=False)
    fruit_predictor = Fruit_Predictor(camera_matrix_file=f"{args.calib_dir}/intrinsic.txt")
    cv_vis = np.ones((480,640,3))* 100


    print("====================Initializing Path Planner====================")

    # Initialize the MapReader object
    map_reader = MapReader(map_fname= args.map, search_list= args.search_list)

    # Get true aruco pos list
    fruits_list, fruits_true_pos, aruco_true_pos = map_reader.load_map_data()

    path_planner = StraightPathPlanner(grid_resolution=0.1, robot_radius=0.1, target_radius=3.7)

    # Load the map and search list
    print(f"Detected fruits: {fruits_list}")
    search_targets = map_reader.read_search_list()
    goals = path_planner.compute_goal_positions(fruits_list, fruits_true_pos, search_targets)
    print(f"Searching in order: {search_targets}")
    print(f"Goal positions: {goals}")

    path_planner.add_obstacles(fruits_true_pos, aruco_true_pos, obstacle_size=0.08)  # Add obstacles
    
    start_x, start_y = 0, 0

    cmd_controller = Command_Controller(ekf=ekf,
                                        aruco=aruco_detector,
                                        control=pictrl,
                                        detector=obj_detector,
                                        predictor=fruit_predictor
                                        )
    
    # Process each target's waypoints
    print("\nAutonomous Driving!")
    goal_count = 0

    # Open file once in write mode
    with open("waypoint.txt", "w") as f:  # Open in write mode to start with an empty file
        for goal in goals:
            print(f"Navigating to {search_targets[goal_count]}")
            cmd_controller.current_goal = search_targets[goal_count]
            print(f"Starting from ({start_x}, {start_y}) to ({goal[0]}, {goal[1]})")

            # Plan the path to the current goal
            waypoint_x, waypoint_y = path_planner.plan_path_based_mode(start_x, start_y, goal[0], goal[1])
            turning_x, turning_y = find_turning_points(waypoint_x, waypoint_y, threshold_angle=15)

            # Reverse the waypoints
            waypoint_x = waypoint_x[::-1]
            waypoint_y = waypoint_y[::-1]

            # Write the goal and waypoints to the file
            f.write(f"Goal {goal_count+1}: {search_targets[goal_count]} at ({goal[0]}, {goal[1]})\n")
            for px, py in zip(waypoint_x, waypoint_y):
                f.write(f"({px},{py})\n")
            # Write turning points to file (optional)
            f.write("Turning Points:\n")
            for tx, ty in zip(turning_x, turning_y):
                f.write(f"{tx},{ty}\n")
            # Plot the waypoints
            plot_waypoints_for_goal(path_planner, waypoint_x, waypoint_y, search_targets[goal_count], show=True)
            
            # Navigate through waypoints, check if each waypoint is a turning point
            for px, py in zip(waypoint_x, waypoint_y):
                is_turning_point = (px, py) in zip(turning_x, turning_y)
                cmd_controller.navigate([px, py], is_turning_point)
                if mt.find_euclidean_distance([goal[0], goal[1]], cmd_controller.get_new_state(drive_meas=Drive(0, 0, 0), period=1)) < 0.3:
                    break

            print("Arrived at destination! Waiting for 5 seconds...")
            time.sleep(5)  # Wait for 5 seconds at the destination

            # Update the start position for the next goal
            current_robot_state = cmd_controller.get_new_state(drive_meas=Drive(0, 0, 0), period=1)
            start_x, start_y = current_robot_state[0][0], current_robot_state[1][0]

            # Increment the goal count
            goal_count += 1

    # Record time
    run_time = time.time() - run_timer
    print(f"\nCompleted in {round(run_time, 2)} seconds")


