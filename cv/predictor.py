import numpy as np

class Fruit_Predictor:
    def __init__(self, camera_matrix_file):
        # Load camera matrix
        self.camera_matrix = np.loadtxt(camera_matrix_file, delimiter=',')
        
        # Define the actual sizes of objects (length x width x height in meters)
        self.object_dimensions = {
            'redapple': [0.076, 0.076, 0.077],  # Red apple
            'greenapple': [0.081, 0.081, 0.073],  # Green apple
            'orange': [0.075, 0.075, 0.072],  # Orange
            'mango': [0.113, 0.067, 0.055],  # Mango
            'capsicum': [0.073, 0.073, 0.088],  # Capsicum
        }

        # This is the prediction of the object.
        # The datatypes would be: list of list
        # 'redapple' : [ [Merged estimations] , [list of estimations] ]
        self.object_prediction = {
            'redapple': [],
            'greenapple': [],
            'orange': [],
            'mango': [],
            'capsicum': [],
        }

        # This is the update flag / known flag for the object
        # If true position is given, it will be toggled to False, and stop updating its position.
        # Else, it would be True, and allow the predictor to update its position.
        self.update_flag = {
            'redapple': True,
            'greenapple': True,
            'orange': True,
            'mango': True,
            'capsicum': True,
        }
    
    def get_fruit_positions_relative_to_camera(self, img_predictions, fruit=None):
        """
        Estimate the positions of multiple fruits relative to the camera based on their bounding boxes and actual sizes.
        If the 'fruit' argument is provided, only the positions for that fruit type will be returned.

        Args:
            img_predictions (list): A list of predictions where each prediction is [class_id, x_center, y_center, width, height].
            fruit (str, optional): The name of the fruit to filter the results (e.g., 'redapple', 'orange').

        Returns:
            list: A list of tuples containing the estimated positions of the fruits relative to the camera (class_id, X, Z) in meters.
        """
        fruit_list = ['redapple', 'greenapple', 'orange', 'mango', 'capsicum']
        estimated_positions = []

        # Extract camera intrinsics
        focal_length_x = self.camera_matrix[0][0]
        focal_length_y = self.camera_matrix[1][1]
        camera_center_x = self.camera_matrix[0][2]
        camera_center_y = self.camera_matrix[1][2]

        for img_prediction in img_predictions:
            # Get the fruit type based on the detected class
            fruit_type = fruit_list[img_prediction[0] - 1]
            
            # If a specific fruit is defined and this fruit doesn't match, skip this iteration
            if fruit and fruit_type != fruit:
                continue
            
            # Extract actual size of the fruit (use height for distance estimation)
            fruit_actual_size = self.object_dimensions[fruit_type][2]  # Use height in meters

            # Extract bounding box details from the prediction
            bbox = self.extract_bounding_box(img_prediction=img_prediction)

            # Bounding box details
            x_center, y_center, width, height = bbox

            # Estimate the distance to the fruit using the height of the bounding box (Z-axis)
            estimated_distance_z = (fruit_actual_size * focal_length_y) / height

            # Calculate the X position relative to the camera (left/right offset)
            estimated_position_x = (x_center - camera_center_x) * estimated_distance_z / focal_length_x

            # Append the estimated position of this fruit
            estimated_positions.append((fruit_type, estimated_position_x, estimated_distance_z))

        return estimated_positions

    def set_ground_truth(self, fruit_list, fruit_pos):
        if fruit_list is None or fruit_pos is None:
            return
        
        # Iterate over provided fruits and their positions
        for idx, fruit in enumerate(fruit_list):
            if fruit in self.object_prediction:
                # Since ground truth exists, disable updates for this fruit
                self.update_flag[fruit] = False
                
                # Update prediction with provided ground truth position
                true_position = fruit_pos[idx]
                self.object_prediction[fruit] = [Estimation(true_pos=true_position)]  # Merged Estimation

    def get_position(self, fruit=None):
        """
        Get the current estimated positions of the fruits.
        If a specific fruit is provided, return the position of that fruit.
        Otherwise, return the positions of all fruits.

        Args:
            fruit (str, optional): The name of the fruit to filter the results (e.g., 'redapple', 'orange').

        Returns:
            dict: A dictionary with fruit names as keys and their positions as values.
        """
        if fruit:
            # Return the position of the specified fruit
            if fruit in self.object_prediction:
                return {fruit: [estimation.get() for estimation in self.object_prediction[fruit]]}
            else:
                return {}
        else:
            # Return the positions of all fruits
            return {key: [estimation.get() for estimation in value] for key, value in self.object_prediction.items()}

    @staticmethod
    def extract_bounding_box(img_prediction):
        # Assuming YOLO format [class_id, x_min, y_min, x_max, y_max]
        class_id, x_min, y_min, x_max, y_max = img_prediction[0:5]

        # Convert (x_min, y_min, x_max, y_max) to (x_center, y_center, width, height)
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min

        return x_center, y_center, width, height
    

class Estimation:
    """
    Estimation class that has three attributes:
    - px: estimated x position
    - py: estimated y position
    - ci: confidence interval
    """
    def __init__(self, robot_pose=None, img_predictions=None, estimation_list=None, true_pos=None):
        """
        Constructor for Estimation class. It allows three types of initialization:
        1. If robot_pose and img_predictions are provided, it will call _predict to compute the prediction.
        2. If estimation_list is provided, it will call _merge_estimations to merge the estimations.
        3. If true_position is provided, it will just assign truepos to px, py, ci
        """
        # If true position is provided, set px, py, ci directly
        if true_pos is not None:
            self.px, self.py, self.ci = true_pos[0], true_pos[1], 1
        # If estimation_list is provided, call _merge_estimations
        elif estimation_list is not None:
            self.px, self.py, self.ci = self._merge_estimations(estimation_list)
        
        # If robot_pose and img_predictions are provided, call _predict
        elif robot_pose is not None and img_predictions is not None:
            self.px, self.py, self.ci = self._predict(robot_pose, img_predictions)
        
        # Otherwise, raise an error for missing arguments
        else:
            raise ValueError("Either provide estimation_list, true_pos, or both robot_pose and img_predictions")

    def _predict(self, robot_pose, img_predictions):
        # Example prediction logic (replace with actual logic)
        return 0.0, 0.0, 1.0  # Example return values
    
    def _merge_estimations(self, estimation_list):
        # Example merge logic for merging multiple estimations
        if not estimation_list:
            return 0.0, 0.0, 1.0

        total_weight = 0
        weighted_sum_x = 0
        weighted_sum_y = 0
        
        for estimation in estimation_list:
            x, y, confidence = estimation
            weighted_sum_x += x * confidence
            weighted_sum_y += y * confidence
            total_weight += confidence
        
        if total_weight == 0:
            return 0.0, 0.0, 1.0
        
        merged_x = weighted_sum_x / total_weight
        merged_y = weighted_sum_y / total_weight
        merged_confidence = total_weight / len(estimation_list)

        return merged_x, merged_y, merged_confidence

    def get(self):
        """
        Get the current estimation values.
        
        Returns:
            tuple: (px, py, ci) The estimated x, y positions and confidence interval.
        """
        return self.px, self.py, self.ci
