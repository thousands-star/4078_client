import numpy as np
from args import args

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
    
    def get_fruit_positions_relative_to_camera(self, img_predictions):
        """
        Estimate the positions of multiple fruits relative to the camera based on their bounding boxes and actual sizes.

        Args:
            img_predictions (list): A list of predictions where each prediction is [class_id, x_center, y_center, width, height].

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
    