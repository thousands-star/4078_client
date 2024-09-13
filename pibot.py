# access each wheel and the camera onboard of pibot
import numpy as np
import requests
import cv2
import time

class PibotControl:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.wheel_vel = [0, 0]
        # We designed a new mode, 0 for manual, 1 for auto
        self.mode = 0
        
    def set_pid(self, use_pid, kp, ki, kd):
        requests.get(f"http://{self.ip}:{self.port}/pid?use_pid="+str(use_pid)+"&kp="+str(kp)+"&ki="+str(ki)+"&kd="+str(kd))

    # Change the robot speed here
    # The value should be between -1 and 1.
    # Note that this is just a number specifying how fast the robot should go, not the actual speed in m/s
    def set_velocity(self, wheel_speed): 
        left_speed = max(min(wheel_speed[0], 1), -1) 
        right_speed = max(min(wheel_speed[1], 1), -1)
        if(self.mode == 0):
            self.wheel_vel = [left_speed, right_speed]
            requests.get(f"http://{self.ip}:{self.port}/move?left_speed="+str(left_speed)+"&right_speed="+str(right_speed))
        return left_speed, right_speed
    
    # Change the displacement here
    # The value here should be in m
    # This function would convert m into optical sensor counting
    def set_displacement(self, displacement):
        if(self.mode == 1):
            unit_disp = round(displacement / 0.00534)
            resp = requests.get(f"http://{self.ip}:{self.port}/disp?left_disp="+str(unit_disp)+"&right_disp="+str(unit_disp))
        return displacement
    
    # Turning around
    # The value here should be in (*degree)
    # The convention is anticlockwise -> positive angle, clockwise -> negative angle
    def set_angle_deg(self, theta):
        if(self.mode == 1):
            radius = 0.08 # radius = baseline/2, in measurement, baseline is 15cm, which is 0.15m.
            if theta < 0:
                radius = 0.085
            unit_disp = (theta / 180 * np.pi * radius) / 0.00534
            
            # We are trying to make it turn, so we have to make different sign on each.
            # Since clockwise is positive angle
            # When clockwise, right have to go in front, left have to go backward.
            resp = requests.get(f"http://{self.ip}:{self.port}/disp?left_disp="+str(-unit_disp)+"&right_disp="+str(unit_disp))
    
        return theta
    
    # Change the displacement here
    # The value here should be in m
    # This function would convert m into optical sensor counting
    def clear_command_queue(self): 
        if(self.mode == 1):
            resp = requests.get(f"http://{self.ip}:{self.port}/disp?left_disp="+str(0)+"&right_disp="+str(0))
        return None
    
    # Switch mode 
    def set_mode(self, mode):
        self.mode = mode
        resp = requests.get(f"http://{self.ip}:{self.port}/mode?driving_mode="+str(mode))

    def get_mode(self):
        return self.mode
        
    def get_image(self):
        try:
            r = requests.get(f"http://{self.ip}:{self.port}/image")
            img = cv2.imdecode(np.frombuffer(r.content,np.uint8), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except (requests.exceptions.ConnectTimeout, requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout) as e:
            print("Image retrieval timed out.")
            img = np.zeros((480,640,3), dtype=np.uint8)
        return img
        
        
# This class stores the wheel velocities of the robot, to be used in the EKF.
class Drive:
    def __init__(self, left_speed, right_speed, dt, left_cov = 1, right_cov = 1):
        self.left_speed = left_speed
        self.right_speed = right_speed
        self.dt = dt
        self.left_cov = left_cov
        self.right_cov = right_cov