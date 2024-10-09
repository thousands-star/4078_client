# estimate the pose of a detected object
import os
import ast
import PIL
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from machinevisiontoolbox import Image
from scipy import stats
import cv2

# Function to read the YOLO bounding boxes from the pred_0.txt file
def read_yolo_predictions(pred_txt_path):
    """
    Reads the YOLO predictions from a text file.
    Returns the formatted_output as a list of [class, x1, y1, x2, y2].
    """
    formatted_output = []
    with open(pred_txt_path, 'r') as f:
        for line in f:
            values = list(map(int, line.strip().split()))
            formatted_output.append(values)  # [class, x1, y1, x2, y2]
    return formatted_output

# use the machinevision toolbox to get the bounding box of the detected object(s) in an image
# Function to handle bounding boxes from YOLO output
def get_bounding_box_from_yolo(formatted_output, object_number, min_size=0):
    """
    Get bounding box from formatted_output from YOLO based on the object number (class id).
    """
    for obj in formatted_output:
        cls, x1, y1, x2, y2 = obj
        
        # Check if the object class matches the one we are looking for
        if cls == object_number:
            width = abs(x2 - x1)
            height = abs(y2 - y1)

            # If the object is too small, we skip it
            if width < min_size or height < min_size:
                print(f"Skipping object {object_number} due to small size: width={width}, height={height}")
                return None  # Return None if the object is too small

            # Calculate center of the bounding box
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Return the bounding box information
            box = [center_x, center_y, int(width), int(height)]
            return box

    return None  # Return None if the object is not found

# Function to extract image info, but using bounding box data from YOLO formatted_output
def get_image_info_from_yolo(file_path, image_poses):
    """
    Extracts object bounding boxes and their associated robot pose info.
    """
    object_lst_box = [[], [], [], [], []]
    object_lst_pose = [[], [], [], [], []]
    completed_img_dict = {}
    formatted_output = read_yolo_predictions(file_path)

    # Loop through each object type (1 to 5)
    for object_num in range(1, 6):  # object labels: 1 to 5
        box = get_bounding_box_from_yolo(formatted_output, object_num)
        if box:
            pose = image_poses[file_path]  
            object_lst_box[object_num - 1].append(box)  # Append bounding box of object
            object_lst_pose[object_num - 1].append(np.array(pose).reshape(3,))  # Robot pose

    # Combine results if more than one object of the same type is detected
    for i in range(5):
        if len(object_lst_box[i]) > 0:
            box = np.stack(object_lst_box[i], axis=1)
            pose = np.stack(object_lst_pose[i], axis=1)
            completed_img_dict[i + 1] = {'object': box, 'robot': pose}

    return completed_img_dict

# estimate the pose of a object based on size and location of its bounding box in the robot's camera view and the robot's pose
def estimate_pose(camera_matrix, completed_img_dict):
    camera_matrix = camera_matrix
    focal_length_x = camera_matrix[0][0]
    focal_length_y = camera_matrix[1][1]
    
    # actual (approximate) sizes of objects
    object_dimensions = []
    redapple_dimensions = [0.076, 0.076, 0.087] #0.082
    object_dimensions.append(redapple_dimensions)
    greenapple_dimensions = [0.081, 0.081, 0.067] # 0.067 -> 0.066
    object_dimensions.append(greenapple_dimensions)
    orange_dimensions = [0.075, 0.075, 0.072] # 0.072 -> 0.069
    object_dimensions.append(orange_dimensions)
    mango_dimensions = [0.113, 0.067, 0.058] # 0.058 -> 0.055
    object_dimensions.append(mango_dimensions)
    capsicum_dimensions = [0.073, 0.073, 0.085] #0.085
    object_dimensions.append(capsicum_dimensions)

    cf = []

    redapple_cf = np.array([0.025,1.0,0]).T
    cf.append(redapple_cf)
    greenapple_cf = np.array([0.05,1.02,0]).T
    cf.append(greenapple_cf)
    orange_cf = np.array([0,1,0]).T
    cf.append(orange_cf)
    mango_cf = np.array([0,0.965,0]).T
    cf.append(mango_cf)
    capsicum_cf = np.array([0.04,1.02,0]).T
    cf.append(capsicum_cf)

    object_list = ['redapple', 'greenapple', 'orange', 'mango', 'capsicum']

    object_pose_dict = {}
    object_dimensions_dict = {}

    
    # for each object in each detection output, estimate its pose
    for object_num in completed_img_dict.keys():
        box = completed_img_dict[object_num]['object'] # [[x],[y],[width],[height]]
        robot_pose = completed_img_dict[object_num]['robot'] # [[x], [y], [theta]]
        true_height = object_dimensions[object_num-1][2]
        true_width = object_dimensions[object_num-1][1]
        
        ######### Replace with your codes #########
        # TODO: compute pose of the object based on bounding box info and robot's pose
        
        # Calculate the distance from the camera to the object using the height of the bounding box
        avg_height = np.mean(box[3])  # Average height from bounding box
        # avg_width = np.mean(box[2])
        coeff = 1
        distance = coeff * (true_height * focal_length_y) / avg_height

        distance_aug = np.array([1,distance,distance**2])

        correction_factor = cf[object_num-1]
        if(object_num == 4):
            #check the height-width ratio, and compensate for the loss
            if(np.mean(box[2]) / avg_height < 1.5):
                print("IF I WERE HERE")
                correction_factor = np.array([0.0565,0.92,0]).T

        
        distance_corrected = distance_aug @ correction_factor

        # distance = (distance + distance_corrected) / 2
        distance = distance_corrected
        


        # Calculate the angle relative to the robot's heading
        # For information: In camera, the x-y-z axis is not same with the robot
        # x-axis : left/right
        # y-axis : up/down(height)
        # z-axis : depth(along the vision sight of camera)

        # Under this circumstances, we need to find the inclined angle of x-axis
        # This is to know whether the object is on our left/right
        # Left: angle = negative value
        # Right: angle = positive value
        camera_x = 250  # x-coordinate of the principal point (optical center)
        box_center_x = np.mean(box[0])  # x-coordinate of the bounding box center
        angle_offset_x = np.arctan((box_center_x - camera_x) / focal_length_x)
        
        X_c = (box_center_x - camera_x) * distance / focal_length_x
        print(f"Distance: {distance}")
        diagonal = np.sqrt(X_c**2 + distance**2)
        # Compute the global position of the object
        # If the object is our right, it is positive
        # This means the angle convention is flipped
        # In robot, anticlockwise is positive, clockwise is negative
        # object to the right in camera is positive, however for robot it is on clockwise direction
        # Therefore, use minus sign.
        robot_x, robot_y, robot_theta = np.mean(robot_pose, axis=1)  # Average robot pose
        
        # robot_length = 0.085
        # robot_x = robot_x + robot_length * np.cos(robot_theta)
        # robot_y = robot_y + robot_length * np.sin(robot_theta)
        print(f"Robot theta:{robot_theta}, angle_offset_x:{angle_offset_x}")
        object_x = robot_x + diagonal* np.cos(robot_theta - angle_offset_x)
        object_y = robot_y + diagonal * np.sin(robot_theta - angle_offset_x)

        # Store the estimated pose
        object_pose_dict[object_list[object_num-1]] = {'x': object_x, 'y': object_y}
        print(f"Class:{object_list[object_num-1]},Xe:{object_x},Ye:{object_y},Box Height:{avg_height},Robot theta:{robot_theta}")
        # object_dimensions_dict[object_list[object_num-1]] = {'width':avg_width, 'height':avg_height}
    

    # print(object_dimensions_dict)
    return object_pose_dict

# merge the estimations of the objects so that there are at most 1 estimate for each object type
def merge_estimations(object_map):
    redapple_est, greenapple_est, orange_est, mango_est, capsicum_est = [], [], [], [], []
    object_est = {}
    num_per_object = 1  # max number of units per object type. We are only using 1 unit per fruit type

    # Combine the estimations from multiple detector outputs
    for f in object_map:
        for key in object_map[f]:
            if key.startswith('redapple'):
                redapple_est.append(np.array(list(object_map[f][key].values()), dtype=float))
            elif key.startswith('greenapple'):
                greenapple_est.append(np.array(list(object_map[f][key].values()), dtype=float))
            elif key.startswith('orange'):
                orange_est.append(np.array(list(object_map[f][key].values()), dtype=float))
            elif key.startswith('mango'):
                mango_est.append(np.array(list(object_map[f][key].values()), dtype=float))
            elif key.startswith('capsicum'):
                capsicum_est.append(np.array(list(object_map[f][key].values()), dtype=float))

    def mode_iqr_mean(estimations):
        if len(estimations) == 0:
            return np.array([0.0, 0.0])

        # Calculate the mode
        mode = stats.mode(estimations, axis=0).mode[0]

        # Calculate the IQR
        Q1 = np.percentile(estimations, 25, axis=0)
        Q3 = np.percentile(estimations, 75, axis=0)
        IQR = Q3 - Q1

        # Filter out the outliers
        filtered_estimations = []
        for est in estimations:
            if np.all((est >= Q1 - 1.5 * IQR) & (est <= Q3 + 1.5 * IQR)):
                filtered_estimations.append(est)
        filtered_estimations = np.array(filtered_estimations)

        # Calculate the mean of the filtered values
        if len(filtered_estimations) > 0:
            return np.mean(filtered_estimations, axis=0)
        else:
            # If all estimations are filtered out, return the mode as the estimation
            return mode

    # Apply the mode_iqr_mean function to each set of fruit estimations
    redapple_avg = geometric_median(redapple_est)
    greenapple_avg = geometric_median(greenapple_est)
    orange_avg = geometric_median(orange_est)
    mango_avg = geometric_median(mango_est)
    capsicum_avg = geometric_median(capsicum_est)

    print("Red_apple_est: ")
    for row in redapple_est:
        print(row)
    print("Green_apple_est: ")
    for row in greenapple_est:
        print(row)
    print("Orange_est: ")
    for row in orange_est:
        print(row)
    print("Mango_est: ")
    for row in mango_est:
        print(row)
    print("Capsicum_est: ")
    for row in capsicum_est:
        print(row)
        
    # Assign the calculated averages to the object_est dictionary
    for i in range(num_per_object):
        object_est['redapple_' + str(i)] = {'x': redapple_avg[0], 'y': redapple_avg[1]}
        object_est['greenapple_' + str(i)] = {'x': greenapple_avg[0], 'y': greenapple_avg[1]}
        object_est['orange_' + str(i)] = {'x': orange_avg[0], 'y': orange_avg[1]}
        object_est['mango_' + str(i)] = {'x': mango_avg[0], 'y': mango_avg[1]}
        object_est['capsicum_' + str(i)] = {'x': capsicum_avg[0], 'y': capsicum_avg[1]}

    return object_est

# Define a function to calculate the median of the estimations
def median_estimate(estimations):
    if len(estimations) > 0:
        return np.median(estimations, axis=0)
    
    return np.array([0.0, 0.0])  # Default to (0,0) if no estimations
def geometric_median(points, tol=1e-7):
    """
    Find the geometric median of a set of points.
    
    Parameters:
    - points: A 2D numpy array where each row represents a point (x, y).
    - tol: Tolerance for convergence.
    
    Returns:
    - The coordinates of the geometric median.
    """
    if len(points) == 0:
        return np.array([0.0, 0.0])
    
    def distance_sum(x):
        return np.sum(np.sqrt(((points - x) ** 2).sum(axis=1)))

    # Initial guess: centroid
    centroid = np.mean(points, axis=0)
    print(centroid)
    result = minimize(distance_sum, centroid, method='COBYLA', tol=tol)
    print(result.x)
    return result.x

if __name__ == "__main__":
    # camera_matrix = np.ones((3,3))/2
    fileK = "{}intrinsic.txt".format('./calibration/param/')
    camera_matrix = np.loadtxt(fileK, delimiter=',')    
    
    # a dictionary of all the saved detector outputs
    # ex : image_poses = { "lab_output/pred_0.png" : [[1.5375283559468054], [1.1314042885549704], [-1.9422776863798767]]}
    image_poses = {}
    with open('lab_output/pred.txt') as fp:
        for line in fp.readlines():
            pose_dict = ast.literal_eval(line)
            image_poses[pose_dict['predfname']] = pose_dict['pose']
    
    # estimate pose of objects in each detector output
    object_map = {}      
    for file_path in image_poses.keys():
        # file path = "lab_output/pred_0.txt", should contain formatted_output from YOLO.detect_single_image(self, np_img)
        completed_img_dict = get_image_info_from_yolo(file_path, image_poses)    #should feed formatted_output(file_path) here
        object_map[file_path] = estimate_pose(camera_matrix, completed_img_dict)

    # merge the estimations of the objects so that there are only one estimate for each object type
    object_est = merge_estimations(object_map)
                     
    # save object pose estimations
    with open('lab_output/objects.txt', 'w') as fo:
        json.dump(object_est, fo, indent=4)
    
    print('Estimations saved!')