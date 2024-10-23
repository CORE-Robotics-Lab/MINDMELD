
import torch
from MINDMELD.Car.DrivingSim.utils.AirSimClient import *
from MINDMELD.Car.config import *
from torchvision import transforms
import time
import requests


class CarEnv:

    def __init__(self, calibration=True, practice=False):
        super(CarEnv, self).__init__()

        self.client = airsim.CarClient()
        self.client.confirmConnection()
        self.car_controls = airsim.CarControls()

        self.goals = np.array(GOALS)
        if calibration:
            self.envs = CALIBRATION_ENVS
        else:
            self.envs = TEST_ENV

        if practice:
            self.envs = PRACTICE_ENV

        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Grayscale(),
                                             transforms.CenterCrop([CROP_SIZE_X, CROP_SIZE_Y])])
        self.prev_im = []

    def reset(self, env_num):
        """
        moves car to specified environment
            Parameters:
                        env_num (int): environment to reset car to
            Returns:
                    state (torch.tensor): tensor defining the current state of the car
                    im: (torch.tensor): tensor image from car

        """
        self.client.enableApiControl(True)
        url = "http://127.0.0.1:5000/"
        requests.post(url, json={"message": "home"})
        time.sleep(3)

        self.client.stopRecording()
        self.step([0, -1])
        time.sleep(3)
        self.step([0, 0])
        print("Environment Number: ", env_num)
        set_pose = self.envs[env_num]['position']
        set_orient = self.envs[env_num]['orientation']

        self.client.simSetVehiclePose(airsim.Pose(set_pose, set_orient), ignore_collison=True)

        state, im = self.get_state()
        return state, im

    def check_collision(self):
        """
              checks if the car has collided with obstacle
                  Returns:
                          (bool): returns True is collision has occured, False otherwise

              """
        collision_info = self.client.simGetCollisionInfo()
        if collision_info.has_collided:
            return True
        return False

    def get_dist(self):
        """
              gets distance between car and the goal
                  Returns:
                          state (torch.tensor): tensor defining the current state of the car
                          im: (torch.tensor): tensor image from car

              """
        car_state, _ = self.get_state()
        position = np.array([car_state[:, 4], car_state[:, 5]])

        dist = np.linalg.norm(position - self.goals)
        return dist.tolist(), position

    def get_state(self):
        """
         gets the current state of the car and the image from airsim. Transforms both into tensors
             Returns:
                     state_list (torch.tensor): tensor defining the current state of the car
                     im: (torch.tensor): tensor image from car

         """
        try:
            img = self.get_image()
            self.prev_im = img
        except Exception:
            img = self.prev_im
        im = self.transform(img)
        im = im.unsqueeze(0)
        car_state = self.client.getCarState()
        pos = car_state.kinematics_estimated.position
        orien = car_state.kinematics_estimated.orientation

        state_list = torch.tensor(
            [self.car_controls.steering, self.car_controls.throttle, self.car_controls.brake, car_state.speed,
             pos.x_val, pos.y_val, orien.w_val, orien.x_val, orien.y_val, orien.z_val]).unsqueeze(0)
        return state_list, im

    def get_image(self):
        """
        Gets the current image frame from Airsim.

        Returns:
            torch tensor of the image dimension 1 x 3 x heigh x width
        """
        image_response = self.client.simGetImages([ImageRequest(1, AirSimImageType.Scene, False, False)])[0]
        image1d = np.fromstring(image_response.image_data_uint8, dtype=np.uint8)
        image_rgba = image1d.reshape(image_response.height, image_response.width, 3)
        return image_rgba

    def step(self, a):
        """
        Appies action, a, for car to take step in evironment
            Parameters:
                        a (list): list of steering and throttle commands to apply to car

        """

        if a[1] <= 0:
            self.car_controls.brake = 4 * abs(float(a[1]))
            self.car_controls.throttle = 0
        else:
            self.car_controls.brake = 0
            self.car_controls.throttle = float(a[1])
        self.car_controls.steering = 1.0 * float(a[0])

        self.client.setCarControls(self.car_controls)
