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
        self.wheel_rad = 6.6e-2/2
        self.counter_dist = 2*np.pi*self.wheel_rad/40
        scale = np.loadtxt("./calibration/param/scale.txt", delimiter=',')
        self.scale = scale
        baseline = np.loadtxt("./calibration/param/baseline.txt", delimiter=',')
        self.baseline = baseline
        self.linear_speed = 0.5
        self.turn_speed = 0.75
        self.dt_right = 0.077753093477052005
        self.dt_left = 0.077553093477052005
        self.resolution = 6
        self.mode = 0
    
    def set_dt(self, dt_left, dt_right):
        self.dt_left, self.dt_right = dt_left, dt_right
       
    def set_lin_pid(self, kp=0.05, ki=0, kd=0.0005):
        requests.get(f"http://{self.ip}:{self.port}/linearpid?kp="+str(kp)+"&ki="+str(ki)+"&kd="+str(kd))
    
    def set_pid(self, use_pid, kp, ki, kd):
        requests.get(f"http://{self.ip}:{self.port}/pid?use_pid="
                     +str(use_pid)+"&kp="+str(kp)+"&ki="+str(ki)+"&kd="+str(kd))
        
    def set_turning_pid(self, kp=0.01, ki=0, kd=0):
        requests.get(f"http://{self.ip}:{self.port}/turnpid?kp="+str(kp)+"&ki="+str(ki)+"&kd="+str(kd))
        
    def set_linear_tol(self, tolerance=6):
        requests.get(f"http://{self.ip}:{self.port}/lineartolerance?tolerance="+str(tolerance))
        
    # Change the robot speed here
    # The value should be between -1 and 1.
    # Note that this is just a number specifying how fast the robot should go, not the actual speed in m/s
    def set_velocity(self, wheel_speed): 
        left_speed = max(min(wheel_speed[0], 1), -1) 
        right_speed = max(min(wheel_speed[1], 1), -1)
        self.wheel_vel = [left_speed, right_speed]
        requests.get(f"http://{self.ip}:{self.port}/move?left_speed="+str(left_speed)+"&right_speed="+str(right_speed))
        return left_speed, right_speed
        
    def get_image(self):
        try:
            r = requests.get(f"http://{self.ip}:{self.port}/image")
            img = cv2.imdecode(np.frombuffer(r.content,np.uint8), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except (requests.exceptions.ConnectTimeout, requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout) as e:
            print("Image retrieval timed out.")
            img = np.zeros((480,640,3), dtype=np.uint8)
        return img
        
    def set_mode(self, mode: int):
        """
        drive mode:
            0: manual control
            1: waypoint navigation
        """
        self.mode = mode
        # print(f"set mode: {mode}")
        requests.get(f"http://{self.ip}:{self.port}/mode?mode="+str(mode))

    def get_mode(self):
        return self.mode
            
    def set_displacement(self, disp: float): 
        left_count = round(disp/self.counter_dist)
        right_count = left_count
        requests.get(f"http://{self.ip}:{self.port}/disp?left_disp="+str(left_count)+"&right_disp="+str(right_count))
        return left_count, right_count
        
    def rounding_on_resolution(self, value:float):
        # Calculate the quotient and remainder
        quotient = value / self.resolution
        remainder = value % self.resolution
        # Determine rounding behavior
        if remainder > (self.resolution / 2):
            return int(quotient) + 1  # Round up
        else:
            return int(quotient)       # Round down
    
    def set_angle_deg(self, angle: float):
        """
        angle set is in degree
        """
        motion_type = "turn left" if angle > 0 else "turn right"
        dt = self.dt_left if angle > 0 else self.dt_right
        loop_num = self.rounding_on_resolution(abs(angle))
        # print(loop_num)
        for _ in range(loop_num):
            requests.get(f"http://{self.ip}:{self.port}/angle?dt="+str(dt)+"&motion="+motion_type)
            time.sleep(0.25)
        return loop_num*self.resolution
        
    
# This class stores the wheel velocities of the robot, to be used in the EKF.
class Drive:
    def __init__(self, left_speed, right_speed, dt, left_cov = 1, right_cov = 1):
        self.left_speed = left_speed
        self.right_speed = right_speed
        self.dt = dt
        self.left_cov = left_cov
        self.right_cov = right_cov
        
        
if __name__ == "__main__":
    bot = PibotControl(1, 2)
    print(bot.baseline)
    print(bot.scale)
