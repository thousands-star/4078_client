import math
import numpy as np
import json
import matplotlib.pyplot as plt
import os

try:
    from path_planner.path_finding import PathPlanner
except ModuleNotFoundError:
    from path_finding import PathPlanner

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
        
        def set_map_fname(map_fname):
            self.map_fname = map_fname
        
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
    
class CommandPlan:
    def __init__(self, mapReader:MapReader, PathPlanner:PathPlanner) -> None:
        print("Path planning initialized...")
        
        self.command_queue= []
        self.replanning_interval = 3
        self.mapReader = mapReader
        self.algorithm = PathPlanner
        self.goal_count = 0
        self.observe_flag = 0
    
    def start(self):
        obstacle_grid_file = "obstacle_grid.txt"
        
        fruits, fruit_positions, aruco_positions = self.mapReader.load_map_data()
        search_list = self.mapReader.read_search_list()

        # Check if the obstacle grid file exists
        if os.path.exists(obstacle_grid_file):
            print("Obstacle grid found, importing...")
            self.algorithm.import_obstacle_grid(obstacle_grid_file)
        else:
            print("Obstacle grid not found, generating a new one...")


            # Add obstacles to the algorithm
            for fruit in fruit_positions:
                self.algorithm.add_obstacle(fruit[0], fruit[1], 0.1)
            for ox, oy in zip(aruco_positions[:, 0], aruco_positions[:, 1]):
                self.algorithm.add_obstacle(ox, oy, 0.1)
            
            # Export the obstacle grid for future runs
            self.algorithm.export_obstacle_grid(obstacle_grid_file)


        self.goals = self.mapReader.compute_goal_positions(fruits, fruit_positions, search_list)

    def plan_command(self, robot_pose, goal):
        robot_x = robot_pose[0]
        robot_y = robot_pose[1]
        robot_theta = robot_pose[2]
        print(str(robot_x) + ":" + str(type(robot_x)))
        print(str(goal[0]) + ":" + str(type(goal[0])))
        waypoint_x, waypoint_y = self.algorithm.plan_path(robot_x, robot_y, goal[0], goal[1])
        est_robot_pose = [robot_x, robot_y, robot_theta]
        planned_point = 0
        for px,py in zip(waypoint_x,waypoint_y):
            if(planned_point < self.replanning_interval):
                theta = MathTools.find_turning_angle([px,py],est_robot_pose)
                if(abs(theta) < 10):
                    pass
                else:
                    self.command_queue.append(['turning', theta])
                    self.command_queue.append(['wait', 0.5])
                    est_robot_pose = [est_robot_pose[0], est_robot_pose[1], est_robot_pose[2]+MathTools.rad(theta)]
                
                disp = MathTools.find_euclidean_distance([px,py],est_robot_pose)
                self.command_queue.append(['forward',disp])
                self.command_queue.append(['wait', 0.5])
                est_robot_pose = [px, py, est_robot_pose[2]]
                planned_point = planned_point + 1
            else:
                break

    def dummy_plan(self, robot_pose, waypoints):
        robot_x = robot_pose[0]
        robot_y = robot_pose[1]
        robot_theta = robot_pose[2]
        print(str(robot_x) + ":" + str(type(robot_x)))
        waypoint_x, waypoint_y = waypoints[0], waypoints[1]
        est_robot_pose = [robot_x, robot_y, robot_theta]
        planned_point = 0
        for px,py in zip(waypoint_x,waypoint_y):
            print(f"Going to {px}, {py}")
            if(planned_point < self.replanning_interval):
                theta = MathTools.find_turning_angle([px,py],est_robot_pose)
                if(abs(theta) < 10):
                    pass
                else:
                    self.command_queue.append(['turning', theta])
                    self.command_queue.append(['wait', 0.5])
                    est_robot_pose = [est_robot_pose[0], est_robot_pose[1], est_robot_pose[2]+MathTools.rad(theta)]
                
                disp = MathTools.find_euclidean_distance([px,py],est_robot_pose)
                self.command_queue.append(['forward',disp])
                self.command_queue.append(['wait', 0.5])
                est_robot_pose = [px, py, est_robot_pose[2]]
                planned_point = planned_point + 1
            else:
                break


    def give_command(self, robot_pose, dummy):
        # This block corresponds to the condition when all goals were found.
        if(self.goal_count >= len(self.goals)):
            return ("Autonomous Driving Finished",['stop', 0])
        else:
            goal = self.goals[self.goal_count]
    
        dist = MathTools.find_euclidean_distance([goal[0],goal[1]],robot_pose)

        # This block corresponds to the condition when we suspect a goal is found
        if(dist < 0.4):
            if self.observe_flag == 0:
                self.command_queue.clear()
                self.command_queue.append(['turning',15])
                self.command_queue.append(['wait',1])
                self.command_queue.append(['turning',-15])
                self.command_queue.append(['wait',1])
                self.command_queue.append(['turning',-15])
                self.command_queue.append(['wait',1])
                self.command_queue.append(['turning',15])
                self.command_queue.append(['wait',1])
                self.observe_flag = 1
                return ("Goal Found Observing",self.command_queue.pop(0))
            if self.observe_flag == 1:
                if(len(self.command_queue) == 0):
                    waiting_time = 5
                    self.command_queue.clear()
                    self.goal_count = self.goal_count + 1
                    self.observe_flag = 0
                    return ("Target Found", ['wait', waiting_time])
                else:
                    return ("Goal Found Observing",self.command_queue.pop(0))

        # This block corresponds to the condition when we dont have any commands in our hand.
        if len(self.command_queue) == 0:
            if self.observe_flag == 0:
                self.command_queue.append(['turning',30])
                self.command_queue.append(['wait',1])
                self.command_queue.append(['turning',-30])
                self.command_queue.append(['wait',1])
                self.command_queue.append(['turning',-30])
                self.command_queue.append(['wait',1])
                self.command_queue.append(['turning',30])
                self.command_queue.append(['wait',1])
                self.observe_flag = 1
                return ("Fixed Interval Observing",self.command_queue.pop(0)) 
            if self.observe_flag == 1:
                if(dummy):
                    self.dummy_plan(robot_pose=robot_pose, waypoints = [[0, 0],[0.4,0]])
                else:
                    self.plan_command(robot_pose=robot_pose, goal=goal)
                self.observe_flag = 0
                return ("Fixed Interval Replanning", self.command_queue.pop(0))
        else:
            return ("Navigating",self.command_queue.pop(0))

        
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
        # print(f"Angle calculated: {theta_w} - {theta_r} = {theta_turn}")
        
        # Calibration for turning time (left turn / right turn)
        # left_turn_time = float( (abs(theta_turn) * left_baseline) / (2 * wheel_speed_turn * scale) )
        # right_turn_time = float( (abs(theta_turn) * right_baseline) / (2 * wheel_speed_turn * scale) )
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
    