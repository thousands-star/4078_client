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

# import SLAM components (M2)
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF 
from slam.robot import Robot
import slam.aruco_sensor as aruco

# import CV components (M3)
sys.path.insert(0,"{}/cv/".format(os.getcwd()))
from cv.detector import *

# import Path Planning components (M4)
from path_planner.command_planning import MapReader
from path_planner.path_finding import PathPlanner
from path_planner.command_planning import MathTools as mt

class Command_Controller:
    def __init__(self, ekf:EKF, aruco:aruco.ArucoSensor, control:PibotControl):
        self.ekf = ekf
        self.aruco_sensor = aruco
        self.control = control
        self.img = None
        # Load the map onto the ekf
        ekf.switch_off_updating()
        self.control.set_mode(1)

    # SLAM with ARUCO markers       
    def get_new_state(self, drive_meas, period = 10):
        self.img = self.control.get_image()
        measurements, self.aruco_img = self.aruco_sensor.detect_marker_positions(self.img)
        self.ekf.predict(drive_meas)
        self.ekf.add_landmarks(measurements)
        self.ekf.update(measurements)
        robot_state = self.ekf.get_state_vector()
        robot_state[2] = mt.clamp_angle(robot_state[2])
        print(f"Robot State: {robot_state[0]},{robot_state[1]},{mt.deg(robot_state[2])}")

        for i in range(period-1):
            self.img = self.control.get_image()
            measurements, self.aruco_img = self.aruco_sensor.detect_marker_positions(self.img)
            self.ekf.predict(Drive(0,0,0))
            self.ekf.add_landmarks(measurements)
            self.ekf.update(measurements)
            time.sleep(0.025)
            # robot_state = self.ekf.get_state_vector()
            # robot_state[2] = mt.clamp_angle(robot_state[2])
            # print(f"Robot State: {robot_state[0]},{robot_state[1]},{mt.deg(robot_state[2])}")

        robot_state = self.ekf.get_state_vector()
        robot_state[2] = mt.clamp_angle(robot_state[2])
        print(f"Robot State: {robot_state[0]},{robot_state[1]},{mt.deg(robot_state[2])}")
        return robot_state
        
    # A feedback mechanism with ekf and camera model and control it to each waypoint.
    def navigate(self, waypoint):
        robot_state = self.ekf.get_state_vector()
        px = waypoint[0]
        py = waypoint[1]
        
        print(f"Waypoint: {px}, {py}")

        theta = mt.find_turning_angle([px,py], robot_state)
        dist = mt.find_euclidean_distance([px,py], robot_state)
        print(f"Distance: {dist}, {theta}")
        while(dist > 0.05):
            while(abs(theta) > 10):
                self.control.set_angle_deg(theta)
                self.ekf.set_var = [0.15, 0.01]
                disp = abs(theta / 180 * np.pi * 0.06)
                # 1 for anticlockwise, -1 for clockwise
                if(theta > 0):
                    left_speed = -0.5
                    right_speed = 0.5
                else:
                    left_speed = 0.5
                    right_speed = -0.5
                drive_meas = Drive(left_speed, right_speed, disp/0.5)
                robot_state = self.get_new_state(drive_meas=drive_meas)
                theta = mt.find_turning_angle([px,py], robot_state)

            left_speed = 0.6
            right_speed = 0.6
            self.ekf.set_var = [0.01, 0.1]
            self.control.set_displacement(0.1)
            disp = 0.1
            drive_meas = Drive(left_speed, right_speed, disp/0.6)
            robot_state = self.get_new_state(drive_meas=drive_meas)
            dist = mt.find_euclidean_distance([px,py], robot_state)
            


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

    # Initialize the Pibot Control
    pictrl = PibotControl(args.ip,args.port)

    # Load camera and wheel calibration parameters
    camera_matrix = np.loadtxt(f"{args.calib_dir}/intrinsic.txt", delimiter=',')
    dist_coeffs = np.loadtxt(f"{args.calib_dir}/distCoeffs.txt", delimiter=',')
    scale = np.loadtxt(f"{args.calib_dir}/scale.txt", delimiter=',')
    baseline = np.loadtxt(f"{args.calib_dir}/baseline.txt", delimiter=',')

    # Initialize the Robot object
    robot = Robot(baseline, scale, camera_matrix, dist_coeffs)

    # Initialize the MapReader object
    map_reader = MapReader(map_fname= args.map, search_list= args.search_list)

    # Get true aruco pos list
    fruits_list, fruits_true_pos, aruco_true_pos = map_reader.load_map_data()
    
    # Initialize EKF with the robot object and provided true map.
    ekf = EKF(robot, truemap="truemap.txt")

    # Initialize the ArUco detector (for updating pose based on markers)
    aruco_detector = aruco.ArucoSensor(robot, marker_length=0.06)

    # Initialise driver
    # drive_controller = MoveToGoalController(robot, K_pw = 1.5, K_pv = 1, ekf = ekf, ppi = ppi, aruco_detector=aruco_detector, aruco_true_pos_list=aruco_true_pos)
    # robot_pose = np.array([0,0,args.initialOrientation*np.pi])

    print("Initializing Path Planner...")

    path_planner = PathPlanner(grid_resolution=0.1, robot_radius = 0.1, target_radius = 5, obstacle_size = 0.1)

    # Load the map and search list
    print(f"Detected fruits: {fruits_list}")
    search_targets = map_reader.read_search_list()
    goals = path_planner.compute_goal_positions(fruits_list, fruits_true_pos, search_targets)
    print(f"Searching in order: {search_targets}")
    print(f"Goal positions: {goals}")

    cmd_controller = Command_Controller(ekf=ekf,aruco=aruco_detector,control=pictrl)

    # Add the markers as obstacles
    for marker in aruco_true_pos:
        path_planner.add_obstacle(marker[0], marker[1], 0.1)
    
    # Add the fruit positions as obstacles
    for fruit in fruits_true_pos:
        path_planner.add_obstacle(fruit[0], fruit[1], 0.1)

    start_x, start_y = 0, 0

    goal_count = 0
    
    # Process each target's waypoints
    print("\nAutonomous Driving!")
    goal_count = 0
    for goal in goals:
        print(f"Navigating to {search_targets[goal_count]}")
        waypoint_x, waypoint_y = path_planner.plan_path(start_x, start_y, goal[0],goal[1])
        # Reverse the waypoints
        waypoint_x = waypoint_x[::-1]
        waypoint_y = waypoint_y[::-1]
        for px,py in zip(waypoint_x,waypoint_y):
            cmd_controller.navigate([px,py])
        print("Arrived at destination! wait for 5 seconds")
        time.sleep(5)
        goal_count = goal_count + 1


    # record time
    run_time = time.time() - run_timer
    print(f"\nCompleted in {round(run_time,2)}")