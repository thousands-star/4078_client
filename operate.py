# basic python packages
import cv2 
import time
import shutil
import os, sys
import numpy as np

# import utility functions
import pygame # python package for GUI
from pibot import Drive
from pibot import PibotControl # access the robot

# import SLAM components (M2)
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
from slam.aruco_sensor import ArucoSensor

# import CV components (M3)
sys.path.insert(0,"{}/cv/".format(os.getcwd()))
from cv.detector import *


class Operate:
    def __init__(self, args):
        
        # Initialise robot controller object
        self.detector_mode = int(args.cv_mode)
        self.pibot_control = PibotControl(args.ip, args.port)
        self.command = {'wheel_speed':[0, 0], # left wheel speed, right wheel speed
                        'save_slam': False,
                        'run_obj_detector': False,                       
                        'save_obj_detector': False,
                        'save_image': False,
                        }
                        
        # TODO: Tune PID parameters here. If you don't want to use PID, set use_pid = 0
        self.pibot_control.set_pid(use_pid=0, kp=1, ki=0, kd=0)
        
        # Create a folder "lab_output" that stores the results of the lab
        self.lab_output_dir = 'lab_output/'
        if not os.path.exists(self.lab_output_dir):
            os.makedirs(self.lab_output_dir)
    
        # Initialise SLAM parameters (M2)
        self.ekf = self.init_ekf(args.calib_dir, args.ip)
        self.aruco_sensor = ArucoSensor(self.ekf.robot, marker_length=0.06) # size of the ARUCO markers (6cm)
        
        # Initialise detector (M3)
        if args.ckpt == "":
            self.obj_detector = None
            self.cv_vis = cv2.imread('ui/8bit/detector_splash.png')
        else:
            if(self.detector_mode == 0):
                self.obj_detector = ResnetDetector(args.ckpt, use_gpu=False)
            elif(self.detector_mode == 1):
                self.obj_detector = YOLODetector(r"C:\Users\Public\ECE4078\project\cv\model\YOLO_best.pt", use_gpu=False)
            self.cv_vis = np.ones((480,640,3))* 100
        
        # Create a folder to save raw camera images after pressing "i" (M3)
        self.raw_img_dir = 'raw_images/'
        if not os.path.exists(self.raw_img_dir):
            os.makedirs(self.raw_img_dir)
        else:
            # Delete the folder and create an empty one, i.e. every operate.py is run, this folder will be empty.
            shutil.rmtree(self.raw_img_dir)
            os.makedirs(self.raw_img_dir)
        

        # Other auxiliary objects/variables      
        self.quit = False
        self.pred_fname = ''
        self.request_recover_robot = False
        self.obj_detector_output = None
        self.ekf_on = False
        self.double_reset_comfirm = 0
        self.image_id = 0
        self.notification = 'Press ENTER to start SLAM'
        self.count_down = 300 # 5 min timer
        self.start_time = time.time()
        self.control_clock = time.time()
        self.img = np.zeros([480,640,3], dtype=np.uint8)
        self.aruco_img = np.zeros([480,640,3], dtype=np.uint8)
        self.obj_detector_pred = np.zeros([480,640], dtype=np.uint8)        
        self.bg = pygame.image.load('ui/gui_mask.jpg')

    # wheel control
    def control(self):    
        if(self.pibot_control.get_mode() == 0):     
            left_speed, right_speed = self.pibot_control.set_velocity(self.command['wheel_speed'])
            dt = time.time() - self.control_clock
            drive_meas = Drive(left_speed, right_speed, dt)
            self.control_clock = time.time()
            drive_meas = Drive(left_speed, right_speed, dt)
        if(self.pibot_control.get_mode() == 1):
            # A Place holder function
            left_speed, right_speed = self.command['wheel_speed'][0] , self.command['wheel_speed'][1]
            if(left_speed > right_speed):
                self.pibot_control.set_angle_deg(-45)
                left_speed = 0.5
                right_speed = -0.5
                theta = 45
                dist = theta / 180 * np.pi * 0.06
            if(right_speed > left_speed):
                self.pibot_control.set_angle_deg(45)
                left_speed = -0.5
                right_speed = 0.5
                theta = 45
                dist = theta / 180 * np.pi * 0.06
            if(left_speed + right_speed > 0):
                self.pibot_control.set_displacement(0.2)
                left_speed = 0.6
                right_speed = 0.6
                dist = 0.2
            if(left_speed + right_speed < 0):
                self.pibot_control.set_displacement(-0.2)
                left_speed = -0.6
                right_speed = -0.6
                dist = 0.2
            if(left_speed == 0 and right_speed == 0):
                dist = 0
            drive_meas = Drive(left_speed, right_speed, dist/abs(left_speed+0.0000001))
            
            
            self.control_clock = time.time()

        return drive_meas
    
    # camera control
    def take_pic(self):
        self.img = self.pibot_control.get_image()
        
    # wheel and camera calibration for SLAM
    def init_ekf(self, calib_dir, ip):
        fileK = os.path.join(calib_dir, 'intrinsic.txt')
        camera_matrix = np.loadtxt(fileK, delimiter=',')
        fileD = os.path.join(calib_dir, 'distCoeffs.txt')
        dist_coeffs = np.loadtxt(fileD, delimiter=',')
        fileS = os.path.join(calib_dir, 'scale.txt')
        scale = np.loadtxt(fileS, delimiter=',')
        fileB = os.path.join(calib_dir, 'baseline.txt')
        baseline = np.loadtxt(fileB, delimiter=',')
        robot = Robot(baseline, scale, camera_matrix, dist_coeffs)
        return EKF(robot)

    # SLAM with ARUCO markers       
    def update_slam(self, drive_meas):
        measurements, self.aruco_img = self.aruco_sensor.detect_marker_positions(self.img)
        if self.request_recover_robot:
            is_success = self.ekf.recover_from_pause(measurements)
            if is_success:
                self.notification = 'Robot pose is successfuly recovered'
                self.ekf_on = True
            else:
                self.notification = 'Recover failed, need >2 landmarks!'
                self.ekf_on = False
            self.request_recover_robot = False
        elif self.ekf_on:
            self.ekf.predict(drive_meas)
            self.ekf.add_landmarks(measurements)
            self.ekf.update(measurements)
            
    # save SLAM map
    def record_data(self):
        if self.command['save_slam']:
            self.ekf.save_map(fname=os.path.join(self.lab_output_dir, 'slam.txt'))
            self.notification = 'Map is saved'
            self.command['save_slam'] = False
        
        # save obj_detector result with the matching robot pose and detector labels
        if self.command['save_obj_detector']:
            if self.obj_detector_output is not None:
                if self.detector_mode == 0:            
                    self.pred_fname = self.obj_detector.write_image(self.obj_detector_output[0], self.obj_detector_output[1], self.lab_output_dir)
                    self.notification = f'Prediction is saved to {operate.pred_fname}'
                elif self.detector_mode == 1:
                    self.pred_fname = self.obj_detector.write_txt(self.obj_detector_output[0], self.obj_detector_output[1], self.lab_output_dir)
                    self.notification = f'Prediction is saved to {operate.pred_fname}'
            else:
                self.notification = f'No prediction in buffer, save ignored'
            self.command['save_obj_detector'] = False

    # using computer vision to detect objects
    def detect_object(self):
        # Hereby, we change the camera output
        # This is the original code
        if self.command['run_obj_detector'] and self.obj_detector is not None:
            self.obj_detector_pred, self.cv_vis = self.obj_detector.detect_single_image(self.img)
            self.command['run_obj_detector'] = False
            self.obj_detector_output = (self.obj_detector_pred, self.ekf.robot.state.tolist())
            if self.detector_mode == 0:
                self.notification = f'{len(np.unique(self.obj_detector_pred))-1} object type(s) detected'
            elif self.detector_mode == 1:
                self.notification = f'{len(self.obj_detector_pred)} object type(s) detected'


    # save raw images taken by the camera after pressing "i"
    def save_image(self):
        if self.command['save_image']:
            image = self.pibot_control.get_image()
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            f_ = os.path.join(self.raw_img_dir, f'img_{self.image_id}.png')
            cv2.imwrite(f_, image)
            self.image_id += 1
            self.command['save_image'] = False
            self.notification = f'{f_} is saved'

    # paint the GUI            
    def draw(self, canvas):
        width, height = 900, 660
        canvas = pygame.display.set_mode((width, height))
        canvas.blit(self.bg, (0, 0))
        text_colour = (220, 220, 220)
        v_pad = 40
        h_pad = 20

        # paint SLAM outputs
        ekf_view = self.ekf.draw_slam_state(res=(520, 480+v_pad), not_pause = self.ekf_on)
        canvas.blit(ekf_view, (2*h_pad+320, v_pad))
        robot_view = cv2.resize(self.aruco_img, (320, 240))
        self.draw_pygame_window(canvas, robot_view, position=(h_pad, v_pad))

        # for object detector (M3)
        detector_view = cv2.resize(self.cv_vis, (320, 240), cv2.INTER_NEAREST)
        self.draw_pygame_window(canvas, detector_view, position=(h_pad, 240+2*v_pad))

        self.put_caption(canvas, caption='SLAM', position=(2*h_pad+320, v_pad))
        self.put_caption(canvas, caption='Detector', position=(h_pad, 240+2*v_pad))
        self.put_caption(canvas, caption='Pi-Bot Cam', position=(h_pad, v_pad))

        notification = TEXT_FONT.render(self.notification, False, text_colour)
        canvas.blit(notification, (h_pad+10, 596))

        time_remain = self.count_down - time.time() + self.start_time
        if time_remain > 0:
            time_remain = f'Count Down: {time_remain:03.0f}s'
        elif int(time_remain)%2 == 0:
            time_remain = "Time Is Up !!!"
        else:
            time_remain = ""
        count_down_surface = TEXT_FONT.render(time_remain, False, (50, 50, 50))
        canvas.blit(count_down_surface, (2*h_pad+320+5, 530))
        return canvas

    @staticmethod
    def draw_pygame_window(canvas, cv2_img, position):
        cv2_img = np.rot90(cv2_img)
        view = pygame.surfarray.make_surface(cv2_img)
        view = pygame.transform.flip(view, True, False)
        canvas.blit(view, position)
    
    @staticmethod
    def put_caption(canvas, caption, position, text_colour=(200, 200, 200)):
        caption_surface = TITLE_FONT.render(caption,
                                          False, text_colour)
        canvas.blit(caption_surface, (position[0], position[1]-25))

    # Keyboard teleoperation
    # For pibot motion, set two numbers for the self.comfmand['wheel_speed']. Eg self.command['wheel_speed'] = [0.6, 0.6]
    # These numbers specify how fast to power the left and right wheels
    # The numbers must be between -1 (full speed backward) and 1 (full speed forward). 0 means stop.
    # Study the code in pibot.py for more information
    def update_keyboard(self):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_a:
                mode = self.pibot_control.get_mode()
                if(mode == 0):
                    self.notification = "The bot is now in autonomous driving mode"
                    self.pibot_control.set_mode(mode = 1)
                if(mode == 1):
                    self.notification = "The bot is now in manual driving mode"
                    self.pibot_control.set_mode(mode = 0)
                pass # TODO
            if event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
                self.command['wheel_speed'] = [0.6, 0.6]
                pass # TODO
            # drive backward
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
                self.command['wheel_speed'] = [-0.6, -0.6]
                pass # TODO
            # turn left
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
                self.command['wheel_speed'] = [-0.35, 0.35]
                pass # TODO
            # drive right
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
                self.command['wheel_speed'] = [0.35, -0.35]
                pass # TODO
            # stop
            elif event.type == pygame.KEYUP or (event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE):
                self.command['wheel_speed'] = [0, 0]
            # run SLAM
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                n_observed_markers = len(self.ekf.taglist)
                if n_observed_markers == 0:
                    if not self.ekf_on:
                        self.notification = 'SLAM is running'
                        self.ekf_on = True
                    else:
                        self.notification = '> 2 landmarks is required for pausing'
                elif n_observed_markers < 3:
                    self.notification = '> 2 landmarks is required for pausing'
                else:
                    if not self.ekf_on:
                        self.request_recover_robot = True
                    self.ekf_on = not self.ekf_on
                    if self.ekf_on:
                        self.notification = 'SLAM is running'
                    else:
                        self.notification = 'SLAM is paused' 
            # save SLAM map
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                self.command['save_slam'] = True
            # reset SLAM map
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                if self.double_reset_comfirm == 0:
                    self.notification = 'Press again to confirm CLEAR MAP'
                    self.double_reset_comfirm +=1
                elif self.double_reset_comfirm == 1:
                    self.notification = 'SLAM Map is cleared'
                    self.double_reset_comfirm = 0
                    self.ekf.reset()
            # Toggle the mode of updating the true map or not.
            # IF it is updating, then the markers will updated from the map
            # Else, it will import from truemap.txt
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_m:
                if(self.ekf_on is False):
                    self.notification = "You have to start slam first!"
                else:
                    mode = self.ekf.get_mode()
                    if(mode == True):
                        self.notification = "SLAM will import from truemap."
                        self.ekf.switch_off_updating()
                    if(mode == False):
                        self.notification = "SLAM will update the marker from EKF."
                        self.ekf.switch_on_updating()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_l:
                if(self.ekf_on is False):
                    self.notification = "You have to start slam first!"
                else:
                    mode = self.ekf.get_lock()
                    if(mode == False):
                        self.notification = "Locked Robot State!"
                        self.ekf.set_lock(True)
                    if(mode == True):
                        self.notification = "Unlock Robot State!"
                        self.ekf.set_lock(False)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_g:
                robotState = self.ekf.robot.state
                x = robotState[0]
                y = robotState[1]
                theta = robotState[2] # In radian
                theta_degree = theta * 180 / np.pi
                while(abs(theta_degree)>360):
                    theta_degree = theta_degree % 360 
                self.notification = f"X:{x} Y:{y} A:{theta_degree}"
            # run object/fruit detector
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                self.command['run_obj_detector'] = True
            # save object detection outputs
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_n:
                self.command['save_obj_detector'] = True
            # capture and save raw image
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_i:
                self.command['save_image'] = True
            # quit
            elif event.type == pygame.QUIT:
                self.quit = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.quit = True
        if self.quit:
            pygame.quit()
            sys.exit()

        
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", metavar='', type=str, default='192.168.137.59')
    parser.add_argument("--port", metavar='', type=int, default=5000)
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    parser.add_argument("--ckpt", default='cv/model/model.best.pt')
    parser.add_argument("--cv_mode", default=0)
    args, _ = parser.parse_known_args()
    
    pygame.font.init() 
    TITLE_FONT = pygame.font.Font('ui/8-BitMadness.ttf', 35)
    TEXT_FONT = pygame.font.Font('ui/8-BitMadness.ttf', 40)
    
    width, height = 700, 660
    canvas = pygame.display.set_mode((width, height))
    pygame.display.set_caption('ECE4078 Lab')
    pygame.display.set_icon(pygame.image.load('ui/8bit/pibot5.png'))
    canvas.fill((0, 0, 0))
    splash = pygame.image.load('ui/loading.png')
    pibot_animate = [pygame.image.load('ui/8bit/pibot1.png'),
                     pygame.image.load('ui/8bit/pibot2.png'),
                     pygame.image.load('ui/8bit/pibot3.png'),
                    pygame.image.load('ui/8bit/pibot4.png'),
                     pygame.image.load('ui/8bit/pibot5.png')]
    pygame.display.update()

    start = False

    counter = 40
    while not start:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                start = True
        canvas.blit(splash, (0, 0))
        x_ = min(counter, 600)
        if x_ < 600:
            canvas.blit(pibot_animate[counter%10//2], (x_, 565))
            pygame.display.update()
            counter += 2

    operate = Operate(args)

    while start:
        operate.update_keyboard()
        operate.take_pic()
        drive_meas = operate.control()
        operate.update_slam(drive_meas)
        operate.record_data()
        operate.save_image()
        operate.detect_object()
        operate.draw(canvas)
        pygame.display.update()