import os
import cv2
import time
import json
import torch
import numpy as np
from args import args
from torchvision import transforms

class ObjectDetector:
    def __init__(self, use_gpu=False):
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.model = None
        self.colour_code = np.array([(220, 220, 220), (128, 0, 0), (155, 255, 70), (255, 85, 0), (255, 180, 0), (0, 128, 0)])  # Example color coding
        self.pred_pose_fname = open(os.path.join('lab_output', 'pred.txt'), 'w')
        self.pred_count = 0
        self.args = args

    def load_weights(self, ckpt_path):
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
            self.model.load_state_dict(ckpt['weights'])
        else:
            print(f'Checkpoint not found, weights are randomly initialized')

    def np_img2torch(self, np_img, _size=(192, 256)):
        preprocess = transforms.Compose([transforms.ToPILImage(),
                                         transforms.Resize(size=_size),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        img = preprocess(np_img).unsqueeze(0)
        if self.use_gpu:
            img = img.cuda()
        return img
    
    def detect_single_image(self, np_img):
        raise NotImplementedError("This method should be implemented in the child class")

    def write_image(self, pred, state, lab_output_dir):
        pred_fname = os.path.join(lab_output_dir, f'pred_{self.pred_count}.png')
        self.pred_count += 1
        cv2.imwrite(pred_fname, pred)
        img_dict = {"pose": state, "predfname": pred_fname}
        self.pred_pose_fname.write(json.dumps(img_dict) + '\n')
        self.pred_pose_fname.flush()
        return f'pred_{self.pred_count - 1}.png'
    
    def write_txt(self, formatted_output, state, lab_output_dir):
        # Create the file path for the .txt file
        pred_fname = os.path.join(lab_output_dir, f'pred_{self.pred_count}.txt')
        
        # Increment the prediction counter
        self.pred_count += 1
        
        # Write the detection results to the .txt file
        with open(pred_fname, 'w') as f:
            for entry in formatted_output:
                cls, x1, y1, x2, y2 = entry
                # Write in the format: cls x1 y1 x2 y2
                f.write(f'{cls} {x1} {y1} {x2} {y2}\n')
        
        # Save the state information
        img_dict = {"pose": state, "predfname": pred_fname}
        self.pred_pose_fname.write(json.dumps(img_dict) + '\n')
        self.pred_pose_fname.flush()
        
        return f'pred_{self.pred_count - 1}.txt'
    
    def visualise_output(self, nn_output):
        raise NotImplementedError("This method should be implemented in the child class")

    def load_weights(self, ckpt_path):
        ckpt_exists = os.path.exists(ckpt_path)
        if ckpt_exists:
            ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
            self.model.load_state_dict(ckpt['weights'])
        else:
            print(f'checkpoint not found, weights are randomly initialised')
            
    @staticmethod
    def np_img2torch(np_img, use_gpu=False, _size=(192, 256)):
        preprocess = transforms.Compose([transforms.ToPILImage(),
                                         transforms.Resize(size=_size),
                                         # transforms.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.3, hue=0.05),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        img = preprocess(np_img)
        img = img.unsqueeze(0)
        if use_gpu: img = img.cuda()
        return img
    

class ResnetDetector(ObjectDetector):
    def __init__(self, ckpt, use_gpu=False):
        super().__init__(use_gpu)
        from res18_skip import Resnet18Skip  # Import the specific ResNet model
        self.model = Resnet18Skip(args)
        if self.use_gpu:
            self.model = self.model.cuda()
        self.load_weights(ckpt)
        self.model.eval()

    def detect_single_image(self, np_img):
        torch_img = self.np_img2torch(np_img)
        tick = time.time()
        with torch.no_grad():
            pred = self.model.forward(torch_img)
            if self.use_gpu:
                pred = torch.argmax(pred.squeeze(), dim=0).detach().cpu().numpy()
            else:
                pred = torch.argmax(pred.squeeze(), dim=0).detach().numpy()
        dt = time.time() - tick
        print(f'Inference Time {dt:.2f}s, approx {1/dt:.2f}fps', end="\r")
        colour_map = self.visualise_output(pred)
        return pred, colour_map

    # This draws a box to see height and width
    def visualise_output(self, nn_output):
        r = np.zeros_like(nn_output).astype(np.uint8)
        g = np.zeros_like(nn_output).astype(np.uint8)
        b = np.zeros_like(nn_output).astype(np.uint8)
        
        # Draw the color-coded segmentation map
        for class_idx in range(0, self.args.n_classes + 1):
            idx = nn_output == class_idx
            r[idx] = self.colour_code[class_idx, 0]
            g[idx] = self.colour_code[class_idx, 1]
            b[idx] = self.colour_code[class_idx, 2]
            
        original_size = nn_output.shape[:2]
        resized_size = (320, 240)
        
        colour_map = np.stack([r, g, b], axis=2)
        colour_map = cv2.resize(colour_map, resized_size, cv2.INTER_NEAREST)
        
        # Calculate scaling factor based on the resize
        x_scale = resized_size[0] / original_size[1]
        y_scale = resized_size[1] / original_size[0]
        
        pt_legend = (10, 160)
        pad = 5
        labels = ['redapple', 'greenapple', 'orange', 'mango', 'capsicum']
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        for i in range(1, self.args.n_classes + 1):
            c = self.colour_code[i]
            
            # Find all contours for the current class
            mask = np.uint8(nn_output == i)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Scale bounding box coordinates to match resized image
                x = int(x * x_scale)
                y = int(y * y_scale)
                w = int(w * x_scale)
                h = int(h * y_scale)
                
                # Calculate the center coordinates
                center_x = x + w // 2
                center_y = y + h // 2
                
                # Draw the bounding box
                colour_map = cv2.rectangle(colour_map, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Prepare the display text with width, height, center_x, and center_y
                dimension_text = f"({w*2}x{h*2})"
                position_text = f"({center_x*2},{center_y*2})"
                
                # Ensure text is placed above the bounding box and within image bounds
                text_x = x
                text_y = max(y - 10, 7)  # Place text 10 pixels above the bounding box
                colour_map = cv2.putText(colour_map, dimension_text, (text_x, text_y), font, 0.35, (255, 255, 255), 1)
                # colour_map = cv2.putText(colour_map, position_text, (text_x, text_y-15), font, 0.35, (255, 255, 255), 1)
            
            # Draw the legend without affecting the size
            legend_rect_size = 10  # Size of the square in the legend
            colour_map = cv2.rectangle(colour_map, pt_legend, (pt_legend[0]+legend_rect_size, pt_legend[1]+legend_rect_size), (int(c[0]), int(c[1]), int(c[2])), thickness=-1)
            colour_map = cv2.putText(colour_map, labels[i-1], (pt_legend[0]+legend_rect_size+pad, pt_legend[1]+legend_rect_size-5), font, 0.4, (0, 0, 0))
            pt_legend = (pt_legend[0], pt_legend[1]+legend_rect_size+pad)
        
        return colour_map
    

class YOLODetector(ObjectDetector):
    def __init__(self, model_path, use_gpu=False):
        super().__init__(use_gpu)
        from ultralytics import YOLO  # Import YOLO model from ultralytics
        self.model = YOLO(model_path)
        if self.use_gpu:
            self.model = self.model.cuda()

    def detect_single_image(self, np_img):
        tick = time.time()
        results = self.model.predict(np_img)
        dt = time.time() - tick
        print(f'Inference Time {dt:.2f}s, approx {1/dt:.2f}fps', end="\r")
        
        formatted_output = []
        # Get bounding boxes and classes
        bboxes, classes = [], []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bboxes.append((x1, y1, x2, y2))
                classes.append(int(box.cls[0]))
                formatted_output.append([int(box.cls[0]+1), x1, y1, x2, y2])
        
        
        # Visualization
        colour_map = self.visualise_output(np_img, bboxes, classes)
        return formatted_output, colour_map

    def visualise_output(self, img, bboxes, classes):
        for bbox, cls in zip(bboxes, classes):
            x1, y1, x2, y2 = bbox
            
            # Ensure the color is in the correct format
            color = tuple(map(int, self.colour_code[cls]))
            
            # Draw the bounding box
            img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Optionally, add the class label above the bounding box
            label = self.model.names[cls]
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        return img