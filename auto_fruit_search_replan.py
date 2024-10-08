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
                 predictor:Fruit_Predictor,
                 detect_enable = [1,1,1]
                 ):
        self.ekf = ekf
        self.aruco_sensor = aruco
        self.control = control
        self.img = None
        self.detector = detector
        self.predictor = predictor
        self.current_goal = 0
        self.current_goal_waypoint = []
        self.detect_enable = detect_enable
        # Load the map onto the ekf
        ekf.switch_off_updating()
        self.control.set_mode(1)
        self.processed_goals = 0
        for i in range(len(self.detect_enable)):
            if(self.detect_enable[i] == 0):
                print(f"Goal {i+1} CV tracking is disable")
            elif(self.detect_enable[i] == 1):
                print(f"Goal {i+1} CV tracking is enable")
            

    # SLAM with ARUCO markers       
    def get_new_state(self, drive_meas, period = 10):
        self.img = self.control.get_image()
        measurements, self.aruco_img = self.aruco_sensor.detect_marker_positions(self.img)
        # Drop camera model when it detected less than 3 marker
        if(len(measurements) < 3):
            measurements = []
        else:
            print("\n3 aruco markers fixing the result!!!\n")

        self.ekf.predict(drive_meas)    # predict ekf
        self.ekf.add_landmarks(measurements)    # add landmarks detected to ekf
        self.ekf.update(measurements)   # update ekf based on landmarks (Correct the predict state)

        robot_state = self.ekf.get_state_vector()   # get the current state vector
        robot_state[2] = mt.clamp_angle(robot_state[2])
    
        for i in range(period-1):
            self.img = self.control.get_image()
            measurements, self.aruco_img = self.aruco_sensor.detect_marker_positions(self.img)
            # Drop camera model when it detected less than 3 marker
            if(len(measurements) < 3):
                measurements = []
            self.ekf.predict(Drive(0,0,0))
            self.ekf.add_landmarks(measurements)
            self.ekf.update(measurements)
            time.sleep(0.05)
            # robot_state = self.ekf.get_state_vector()
            # robot_state[2] = mt.clamp_angle(robot_state[2])
            # print(f"Robot State: {robot_state[0]},{robot_state[1]},{mt.deg(robot_state[2])}")

        robot_state = self.ekf.get_state_vector()
        robot_state[2] = mt.clamp_angle(robot_state[2])
        print(f"After {period} steps EKF: Robot State: {robot_state[0]},{robot_state[1]},{mt.deg(robot_state[2])}")
        return robot_state

    # A feedback mechanism with ekf and camera model and control it to each waypoint.
    def navigate(self, waypoint, is_turning_point=False, is_final_turning_point=False):
        robot_state = self.ekf.get_state_vector()
        px = waypoint[0]
        py = waypoint[1]

        print(f"Waypoint: {px}, {py}")

        # Calculate the distance and angle to the next waypoint
        theta = mt.find_turning_angle([px, py], robot_state)
        dist = mt.find_euclidean_distance([px, py], robot_state)
        print(f"Distance: {dist}, Angle: {theta}")

        # Motion Model turning
        if is_turning_point:
            # Prioritize precise rotation at turning points
            print("\nAt a turning point, rotating precisely...")
            print(f"Correcting angle at turning point: {theta}")
            while abs(theta) > 3:  # Adjust this threshold for fine rotation control
                if theta > 90:
                    theta = 90
                elif theta < -90:
                    theta = -90

                time.sleep(0.5)
                # Rotate to correct the angle
                self.control.set_angle_deg(theta)
                self.ekf.set_var = [0.1, 0.01]
                drive_meas = self.get_drive_from_theta(theta)
                robot_state = self.get_new_state(drive_meas=drive_meas)
                theta = mt.find_turning_angle([px, py], robot_state)
                print(f"Correcting angle at turning point: {theta}")
            print("Finish turning point turning! \n")
        
        # CV correcting angle and distance.
        if is_final_turning_point and self.detect_enable[self.processed_goals] == 1:
            # get the angle between final turning point and goal.
            tp_g_theta = mt.find_turning_angle(waypoint=self.current_goal_waypoint, robot_pose=robot_state)
            print(f"desired angle:{tp_g_theta}")
            resp = self.cv_correction(desired_angle=tp_g_theta)
            if resp is True:
                print("CV was triggered to correct the final step")
                # Get the updated robot state and fruit position
                robot_state = self.ekf.get_state_vector()
                self.img = self.control.get_image()  # Get a new image
                cv2.imwrite("forward_first.png", self.img)
                image_preds, _ = self.detector.detect_single_image(self.img)
                fruit_pos = self.predictor.get_fruit_positions_relative_to_camera(image_preds, fruit=self.current_goal)

                if fruit_pos:
                    # Calculate the distance between robot and fruit
                    x_dist, z_dist = fruit_pos[0][1], fruit_pos[0][2]
                    dist_to_fruit = (x_dist ** 2 + z_dist ** 2) ** 0.5
                    
                    print(f"Distance to fruit: {dist_to_fruit} meters")
                    first_detect = dist_to_fruit
                    forward = 0
                    # Move towards the fruit
                    while dist_to_fruit > 0.3 and (first_detect - forward*0.15) > 0.3:
                        angle = np.arctan(x_dist / z_dist) * 180 / np.pi
                        self.control.set_angle_deg(-angle)
                        print(f"Turning {angle:.2f} towards the fruit.")
                        dis_step = min(0.15, dist_to_fruit / 2)
                        print(f"Moving {dis_step:.2f} towards the fruit. Current distance: {dist_to_fruit:.2f}m")
                        forward += 1

                        left_speed = right_speed = 0.6
                        self.control.set_displacement(dis_step)
                        drive_meas = Drive(left_speed, right_speed, dis_step / 0.6)

                        # Update robot state and recalculate distance
                        robot_state = self.get_new_state(drive_meas=drive_meas)
                        self.img = self.control.get_image()  # Get a new image for fruit detection
                        cv2.imwrite("forwarding.png", self.img)
                        image_preds, _ = self.detector.detect_single_image(self.img)
                        fruit_pos = self.predictor.get_fruit_positions_relative_to_camera(image_preds, fruit=self.current_goal)
                        
                        if not fruit_pos:
                            dis_step = max((first_detect - forward*0.15) - 0.3, dist_to_fruit - 0.45)
                            self.control.set_displacement(dis_step)
                            drive_meas = Drive(left_speed, right_speed, dis_step / 0.6)
                            robot_state = self.get_new_state(drive_meas=drive_meas)
                            break

                        x_dist, z_dist = fruit_pos[0][1], fruit_pos[0][2]
                        dist_to_fruit = (x_dist ** 2 + z_dist ** 2) ** 0.5  # Recalculate distance to the fruit
                        
                    print("Arrived within 0.05m of the fruit. Stopping.")
                    if(self.current_goal == 1):
                        dis_step = 0.15
                        self.control.set_displacement(dis_step)
                        drive_meas = Drive(left_speed, right_speed, dis_step / 0.6)
                        robot_state = self.get_new_state(drive_meas=drive_meas)
                    return  # Exit the function after reaching the fruit
            else:
                print("CV correction failed. Continuing with regular navigation.")
        elif is_final_turning_point and self.detect_enable[self.processed_goals] == 0:
            pass

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
    
    def get_drive_from_theta(self, theta):
        radius = 0.06
        disp = abs(mt.rad(theta) * radius)
        if theta > 0:
            left_speed = -0.8   # based on the listen_2 turning speed
            right_speed = 0.8
        else:
            left_speed = 0.8
            right_speed = -0.8
        drive_meas = Drive(left_speed, right_speed, disp / 0.8)
        return drive_meas


    def cv_correction(self, desired_angle = 0):
        # Get the current image and save it for debugging purposes
        self.img = self.control.get_image()
        cv2.imwrite("first_capture.png", self.img)

        # Detect fruits in the image
        image_preds, _ = self.detector.detect_single_image(self.img)
        pos = self.predictor.get_fruit_positions_relative_to_camera(image_preds, fruit=self.current_goal)

                # Variables for limiting search to ±90 degrees
        angle_increment = 18  # Degrees to turn on each attempt
        max_angle = 72  # Maximum angle to turn (either +90 or -90 degrees)
        angle_turned = 0  # Track the total angle turned
        total_angle_turned = 0
        total_max_angle = max_angle * 4
        direction = 1  # Start by turning right (positive direction)

        while len(pos) == 0 and abs(angle_turned) <= max_angle:
            print(f"No fruit detected, turning the robot by {direction * angle_increment} degrees.")

            # Turn the robot by the current increment in the given direction
            self.control.set_angle_deg(direction * angle_increment)
            drive_meas = self.get_drive_from_theta(direction * angle_increment)
            self.get_new_state(drive_meas=drive_meas, period = 1)

            angle_turned += direction * angle_increment  # Update total angle turned
            total_angle_turned += angle_increment
            time.sleep(0.5)

            # Get a new image after rotating
            self.img = self.control.get_image()
            image_preds, _ = self.detector.detect_single_image(self.img)
            pos = self.predictor.get_fruit_positions_relative_to_camera(image_preds, fruit=self.current_goal)

            # If the fruit is detected after rotating, break the loop
            if len(pos) > 0 or total_angle_turned >= total_max_angle:
                break

            # If we've reached +90 degrees, switch to turning in the opposite direction
            if angle_turned >= max_angle:
                direction = -1  # Switch to turning left
            elif angle_turned <= -max_angle:
                direction = 1  # Switch back to turning right

        # If no fruit was detected after all attempts, return False
        if len(pos) == 0:
            print("Fruit not detected after maximum attempts, stopping.")
            return False

        # If fruit is detected, continue with the correction logic
        pos = pos[0]
        x_dist = pos[1]
        z_dist = pos[2]
        dist = (x_dist ** 2 + z_dist ** 2) ** 0.5
        cv_angle = np.arctan(x_dist / z_dist) * 180 / np.pi

        # Perform angle correction if the distance is greater than 0.6 meters
        last_angle_turned = 0  # To store the last turned angle in case we need to recover
        timeout = 0
        if dist > 0.5:
            tout = 0
            # Correct the angle as long as the angle is more than 7 degrees
            while abs(cv_angle + desired_angle) > 7:
                print(f"CV correcting angle: {cv_angle+desired_angle}")
                self.control.set_angle_deg(-(cv_angle+desired_angle))  # Adjust the robot's angle
                # drive_meas = self.get_drive_from_theta(-(cv_angle+desired_angle))
                # self.get_new_state(drive_meas=drive_meas, period = 1)

                time.sleep(0.5)
                last_angle_turned = -(cv_angle+desired_angle)  # Remember the angle turned in this step
                
                # Reinitialize pos and timeout for this iteration
                pos = []
                timeout = 0

                # Retry fruit detection with a timeout of 3 attempts
                while len(pos) == 0 and timeout < 3:
                    self.img = self.control.get_image()  # Get a new image
                    image_preds, _ = self.detector.detect_single_image(self.img)
                    pos = self.predictor.get_fruit_positions_relative_to_camera(image_preds, fruit=self.current_goal)

                    time.sleep(0.5)  # Small delay
                    timeout += 1

                if len(pos) > 0:
                    pos = pos[0]
                    x_dist = pos[1]
                    z_dist = pos[2]
                    dist = (x_dist ** 2 + z_dist ** 2) ** 0.5
                    cv_angle = np.arctan(x_dist / z_dist) * 180 / np.pi
                else:
                    # If fruit is lost after turning, recover the last turned angle
                    print("Failed to detect fruit, recovering the previous angle.")
                    self.control.set_angle_deg(-last_angle_turned*2)  # Recover the last angle turned
                    drive_meas = self.get_drive_from_theta(-last_angle_turned*2)
                    self.get_new_state(drive_meas=drive_meas, period = 1)
                    if(tout > 1):
                        break
                    tout = tout + 1
                    continue


            # Log the corrected robot state
            print(f"\nCV corrected state: {self.ekf.robot.state[0]},{self.ekf.robot.state[1]},{self.ekf.robot.state[2]}")
        return True
                

###### 4.3 ##########################################################        
    def detect_new_fruit(self, fruits_list, detected_fruit_types):
        """
        Detects new fruits and checks their type to avoid adding the same fruit type multiple times.

        Parameters:
        - fruits_list: List of known fruit types in the map.
        - detected_fruit_types: Set of fruit types already added to the map.

        Returns:
        - List of new fruits [(fruit_type, global_position)].
        """
        # Get the current image and detect fruits
        self.img = self.control.get_image()
        detected_fruits, _ = self.detector.detect_single_image(self.img)
        cv2.imwrite("detect_new_fruit.png", self.img)
        # Get the robot's current position and orientation from the EKF
        robot_state = self.ekf.get_state_vector()
        robot_x, robot_y, robot_theta = robot_state[0], robot_state[1], robot_state[2]

        new_fruits = []

        # Get the fruit positions relative to the camera for all detected fruits (no filter by type)
        fruit_positions = self.predictor.get_fruit_positions_relative_to_camera(detected_fruits)

        if fruit_positions:
            for fruit_type, x_dist, z_dist in fruit_positions:
                # Skip if the fruit type is already in the map or detected earlier
                if fruit_type in fruits_list:
                    #print(f"Skipping {fruit_type}, already in the truemap.")
                    continue

                if fruit_type in detected_fruit_types:
                    #print(f"Skipping {fruit_type}, already detected at {detected_fruit_types[fruit_type]}.")
                    continue

                # Calculate the distance to the fruit
                dist_to_fruit = (x_dist ** 2 + z_dist ** 2) ** 0.5
                angle_offset = np.arctan2(x_dist, z_dist) * 180 / np.pi

                # Convert the fruit's relative coordinates to global coordinates
                fruit_global_x = robot_x + dist_to_fruit * np.cos(robot_theta - np.radians(angle_offset))
                fruit_global_y = robot_y + dist_to_fruit * np.sin(robot_theta - np.radians(angle_offset))
                fruit_global_pos = (fruit_global_x, fruit_global_y)

                # Add the new fruit to the list
                print(f"New fruit detected: {fruit_type} at global coordinates {fruit_global_pos}")
                new_fruits.append((fruit_type, fruit_global_pos))  # Store fruit type and its global position

        # Return the list of new fruit coordinates and types (for replanning if necessary)
        return new_fruits if new_fruits else []
    
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
    plt.gca().set_xticks(np.arange(-1.6, 1.6, 0.4))
    plt.gca().set_yticks(np.arange(-1.6, 1.6, 0.4))
    plt.xlim(-1.6, 1.6)
    plt.ylim(-1.6, 1.6)

    # Add labels and a title to the plot
    plt.title(f"Path to {search_target}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)

    # Show the plot if required
    plt.show()


def find_turning_points(path_x, path_y, intermediate_flags, threshold_angle=15, min_distance=0.1):
    """
    Find turning points based on the angle between three consecutive points,
    and also include intermediate points.
    
    path_x: List of x-coordinates of the waypoints.
    path_y: List of y-coordinates of the waypoints.
    intermediate_flags: List of flags indicating whether a point is intermediate or original.
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
        return abs((p2[1] - p1[1]) * (p3[0] - p2[0]) - (p3[1] - p2[1]) * (p2[0] - p1[0])) < tolerance

    # Iterate over the waypoints and find turning points
    for i in range(1, len(path_x) - 1):
        x1, y1 = path_x[i - 1], path_y[i - 1]
        x2, y2 = path_x[i], path_y[i]
        x3, y3 = path_x[i + 1], path_y[i + 1]

        # Always consider intermediate points as turning points
        if intermediate_flags[i]:
            turning_points_x.append(x2)
            turning_points_y.append(y2)
            continue  # Skip further checks for intermediate points

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


def add_intermediate_points(waypoint_x, waypoint_y, max_distance=1.2):
    new_waypoint_x = [waypoint_x[0]]  # Start with the first waypoint
    new_waypoint_y = [waypoint_y[0]]
    intermediate_flags = [False]  # False indicates that it's an original point, not intermediate
    
    for i in range(1, len(waypoint_x)):
        # Calculate the distance between the current and previous waypoint
        dist = np.sqrt((waypoint_x[i] - waypoint_x[i-1])**2 + (waypoint_y[i] - waypoint_y[i-1])**2)
        
        if dist > max_distance:
            # Find the direction vector between the two waypoints
            direction_x = (waypoint_x[i] - waypoint_x[i-1]) / dist
            direction_y = (waypoint_y[i] - waypoint_y[i-1]) / dist
            
            # Add intermediate points
            new_x = waypoint_x[i-1] + direction_x * (dist/2)
            new_y = waypoint_y[i-1] + direction_y * (dist/2)
            new_waypoint_x.append(new_x)
            new_waypoint_y.append(new_y)
            intermediate_flags.append(True)  # Mark this as an intermediate point
        
        # Append the original waypoint after adding any necessary intermediate points
        new_waypoint_x.append(waypoint_x[i])
        new_waypoint_y.append(waypoint_y[i])
        intermediate_flags.append(False)  # Mark this as an original point

    return new_waypoint_x, new_waypoint_y, intermediate_flags


def replan_path(path_planner, current_position, new_fruits, current_goal, fruits_list, fruits_true_pos, aruco_true_pos, f):
    print(f"Replanning path from position {current_position} due to new fruit detection...")

    # Reset the obstacles, keeping only the static ones (like walls)
    path_planner.reset_obstacles()

    # Now add the new fruits to the fruits list and fruits_true_pos
    for fruit_type, fruit_global_pos in new_fruits:
        print(f"Adding new obstacle: {fruit_type} at {fruit_global_pos}")
        fruits_list.append(fruit_type)  # Add fruit type to the list
        fruit_global_pos_flat = np.hstack(fruit_global_pos)  # Flatten the fruit_global_pos
        fruits_true_pos = np.append(fruits_true_pos, [fruit_global_pos_flat], axis=0)  # Append the flattened array

    # Re-add obstacles including the updated fruit positions
    path_planner.add_obstacles(fruits_true_pos, aruco_true_pos, obstacle_size=0.08)

    # Recalculate the path from the current position to the current goal
    current_x = float(current_position[0][0])  # Extract the scalar value for x
    current_y = float(current_position[1][0])  # Extract the scalar value for y
    goal_x, goal_y = current_goal[0], current_goal[1]

    # Plan a new path with the updated obstacles
    updated_path_x, updated_path_y = path_planner.plan_path_based_mode(current_x, current_y, goal_x, goal_y)

    # # Ensure the current EKF position is included as the first point in the new path
    # if not (updated_path_x[0] == current_x and updated_path_y[0] == current_y):
    #     updated_path_x.insert(0, current_x)
    #     updated_path_y.insert(0, current_y)

    # Write the replanned waypoints to waypoint.txt

    return updated_path_x, updated_path_y

# main loop
if __name__ == "__main__":
    # argument parsing for command-line execution
    parser = argparse.ArgumentParser("Fruit searching")
    parser.add_argument("--map", type=str, default='truemap_cv.txt')
    parser.add_argument("--search_list", type=str, default='search_list.txt')
    parser.add_argument("--ip", metavar='', type=str, default='192.168.0.104')
    parser.add_argument("--port", metavar='', type=int, default=5000)
    parser.add_argument("--calib_dir", type=str, default="calibration/param/", help="Directory containing calibration parameters")
    parser.add_argument("--yolo_model", default=r'C:\Users\Asus\Desktop\Monash year3\ece4078\milestone1\cv\model\best.pt', help="YOLO model file for object detection")
    
    parser.add_argument("--grid_resolution", default = 0.1, type = float)
    parser.add_argument("--robot_radius", default = 0.15, type = float)
    parser.add_argument("--target_radius", default = 4, type = float)
    parser.add_argument("--obstacle_size", default = 0.08, type = float)

    detection_flag = [1,1,1]
    args, _ = parser.parse_known_args()

    # start timer
    run_timer = time.time()

    # M2 Initialization
    # Initialize the Pibot Control
    print("--------------------Initializing ROBOT & EKF--------------------")
    # Remember tune the motion model
    #kp = settling time,ki = error, kd =overshoot
    pictrl = PibotControl(args.ip,args.port)
    pictrl.set_lin_pid(kp=1.75, ki = 0.05, kd = 0.01)
    pictrl.set_turning_pid(kp= 0.1,ki = 0.005,kd = 0.001)

    pictrl.set_mode(mode=1)
    pictrl.set_dt(dt_left=0.077753093477052005,dt_right = 0.077553093477052005)
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
    # obj_detector = ResnetDetector(r"cv\model\model.best.pt",use_gpu=False)
    fruit_predictor = Fruit_Predictor(camera_matrix_file=f"{args.calib_dir}/intrinsic.txt", fruit_detector=obj_detector)
    cv_vis = np.ones((480,640,3))* 100


    print("====================Initializing Path Planner====================")
    # Initialize Path Planner (M4)
    # Initialize the MapReader object
    map_reader = MapReader(map_fname= args.map, search_list= args.search_list)

    # Get true aruco pos list
    fruits_list, fruits_true_pos, aruco_true_pos = map_reader.load_map_data()

    path_planner = DiagonalPathPlanner(grid_resolution=args.grid_resolution, robot_radius=args.robot_radius, target_radius=args.target_radius)

    # Load the map and search list
    print(f"Detected fruits: {fruits_list}")
    search_targets = map_reader.read_search_list()
    goals = path_planner.compute_goal_positions(fruits_list, fruits_true_pos, search_targets)
    print(f"Searching in order: {search_targets}")
    print(f"Goal positions: {goals}")

    path_planner.add_obstacles(fruits_true_pos, aruco_true_pos, obstacle_size=0.08)  # Add obstacles
    
    start_x, start_y = 0, 0

    # Path Follower Initialized.
    cmd_controller = Command_Controller(ekf=ekf,
                                        aruco=aruco_detector,
                                        control=pictrl,
                                        detector=obj_detector,
                                        predictor=fruit_predictor,
                                        detect_enable= detection_flag
                                        )
    
    # Process each target's waypoints
    print("\n====================Path Planner Started to Search for Path====================\n")
    goal_count = 0

    # Open file once in write mode
    with open("waypoint.txt", "w") as f:  # Open in write mode to start with an empty file
        for goal in goals:
            print(f"Navigating to {search_targets[goal_count]}")
            cmd_controller.current_goal = search_targets[goal_count]
            cmd_controller.current_goal_waypoint = [goal[0],goal[1]]
            print(f"Starting from ({start_x}, {start_y}) to ({goal[0]}, {goal[1]})")

            # Plan the path to the current goal
            waypoint_x, waypoint_y = path_planner.plan_path_based_mode(start_x, start_y, goal[0], goal[1])
            # Add intermediate points where necessary
            waypoint_x, waypoint_y, intermediate_flags = add_intermediate_points(waypoint_x, waypoint_y, max_distance=1.2)

            # Find turning points, including intermediate points
            turning_x, turning_y = find_turning_points(waypoint_x, waypoint_y, intermediate_flags, threshold_angle=15)

            # Reverse the waypoints
            waypoint_x = waypoint_x[::-1]
            waypoint_y = waypoint_y[::-1]

            # Reverse the turning points
            turning_x = turning_x[::-1]
            turning_y = turning_y[::-1]

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

            # Navigate through waypoints and check for new fruits
            i = 0

            detected_fruit_types = set()  # Initialize at the start of the program

            while i < len(waypoint_x):
                px, py = waypoint_x[i], waypoint_y[i]
                is_turning_point = (px, py) in zip(turning_x, turning_y)

                # Check if the current waypoint is the final turning point
                is_final_turning_point = (px, py) == (turning_x[-1], turning_y[-1])

                if is_final_turning_point:
                    print(f"Final Point: {px},{py}")

                # Detect new fruits at this waypoint
                for attempt in range(10):
                    new_fruits = cmd_controller.detect_new_fruit(fruits_list, detected_fruit_types)
                    if new_fruits:
                        break
                    else:
                        time.sleep(0.1)

                if new_fruits:
                    print(f"Detected new fruit: {new_fruits}. Triggering replanning.")
                    current_position = cmd_controller.ekf.get_state_vector()

                    # Update detected fruit types (to prevent repeated fruits)
                    for fruit, pos in new_fruits:
                        detected_fruit_types.add(fruit)

                    # Replan the path based on the newly detected fruit
                    waypoint_x, waypoint_y = replan_path(path_planner, current_position, new_fruits, goal, fruits_list, fruits_true_pos, aruco_true_pos, f)

                    # Add intermediate points where necessary after replanning
                    waypoint_x, waypoint_y, intermediate_flags = add_intermediate_points(waypoint_x, waypoint_y, max_distance=1.2)

                    # Find turning points again after replanning
                    turning_x, turning_y = find_turning_points(waypoint_x, waypoint_y, intermediate_flags, threshold_angle=15)

                    # Reverse the waypoints again
                    waypoint_x = waypoint_x[::-1]
                    waypoint_y = waypoint_y[::-1]

                    # Reverse the turning points
                    turning_x = turning_x[::-1]
                    turning_y = turning_y[::-1]

                    # Write the goal and waypoints to the file
                    f.write(f"Replanning to Goal {goal_count+1}: {search_targets[goal_count]} at ({goal[0]}, {goal[1]})\n")
                    for px, py in zip(waypoint_x, waypoint_y):
                        f.write(f"({px},{py})\n")

                    # Write turning points to file (optional)
                    f.write("Replanned turning Points:\n")
                    for tx, ty in zip(turning_x, turning_y):
                        f.write(f"{tx},{ty}\n")

                    # Plot the waypoints
                    print("After replanning:")
                    plot_waypoints_for_goal(path_planner, waypoint_x, waypoint_y, search_targets[goal_count], show=True)

                    # Reset the index to restart navigation from the updated path
                    i = 0
                    continue  # Restart the loop with updated waypoints

                # Navigate through waypoints
                cmd_controller.navigate([px, py], is_turning_point, is_final_turning_point)

                # Check if close enough to the goal and break if so
                if mt.find_euclidean_distance([goal[0], goal[1]], cmd_controller.get_new_state(drive_meas=Drive(0, 0, 0), period=1)) < 0.3:
                    break

                # Move to the next waypoint
                i += 1

            print("Arrived at destination! Waiting for 1 seconds...")
            time.sleep(1)  # Wait for 5 seconds at the destination

            # Update the start position for the next goal
            current_robot_state = cmd_controller.get_new_state(drive_meas=Drive(0, 0, 0), period=1)
            start_x, start_y = current_robot_state[0][0], current_robot_state[1][0]

            # Check if the updated position is inside an obstacle
            if not path_planner.is_node_valid(path_planner.Node(path_planner.get_xy_index(start_x, path_planner.grid_min_x), 
                                                                path_planner.get_xy_index(start_y, path_planner.grid_min_y), 
                                                                cost=0.0, parent_idx=-1)):
                print(f"Warning: The new position ({start_x}, {start_y}) is inside an obstacle.")
                print("Setting the last valid waypoint as the new starting position.")
                
                # Set the last waypoint as the new start position
                start_x, start_y = waypoint_x[-1], waypoint_y[-1]
                print(f"New starting position set to ({start_x}, {start_y})")

            # Increment the goal count
            goal_count += 1

        # Record time
        run_time = time.time() - run_timer
        print(f"\nCompleted in {round(run_time, 2)} seconds")

