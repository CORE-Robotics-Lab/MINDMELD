#import h5py as h5
from torch.utils import data
import pickle
import numpy as np
import torch

class CarWheelDataset(data.Dataset):
    def __init__(self,car_path, wheel_path,transform=None):
        car_data = []
        wheel_data = []
        with open(car_path, "rb") as f:
            car_data.append(pickle.load(f, encoding='bytes'))
        with open(wheel_path, "rb") as f:
            wheel_data.append(pickle.load(f, encoding='bytes'))

        angle = []
        throttle = []
        brake = []
        images = []
        states = []
        actions = []
        for i in range(len(car_data)):
            wheel = wheel_data[i]
            car = car_data[i]

            for i in range(len(wheel)):
                angle.append(wheel[i]['wheel'])
                throttle.append(wheel[i]['throttle'])
                brake.append(wheel[i]['brake'])
                actions.append(torch.FloatTensor([wheel[i]['wheel'], wheel[i]['throttle']-wheel[i]['brake']]))
            for i in range(len(car)):
                states.append(self.flatten_state(car[i]['data'], actions[i]))
                images.append(car[i]['image'])

        self.labels = actions
        self.data = states
        self.im = images

        self.transform=transform

    def __getitem__(self, index):
        data=self.data[index]
        im=self.im[index]
        if self.transform is not None:
            im=self.transform(im)
        return im,data,self.labels[index]

    def __len__(self):
        return len(self.labels)


    def flatten_state(self, state, wheel_state):#, prev_wheel_state):
        # TODO: zero-mean normalization
        flattened = []
        #flattened.append(state.gear / 10)
        #flattened.append(state.handbrake)
        k_e = state.kinematics_estimated
        flattened += self.flatten_vector(k_e.angular_acceleration)
        flattened += self.flatten_vector(k_e.angular_velocity)
        flattened += self.flatten_vector(k_e.linear_acceleration)
        flattened += self.flatten_vector(k_e.linear_velocity)
        flattened.append(k_e.orientation.w_val)
        flattened.append(k_e.orientation.x_val)
        flattened.append(k_e.orientation.y_val)
        flattened.append(k_e.orientation.z_val )
        flattened += self.flatten_vector(k_e.position)
        #flattened.append(state.maxrpm / 7500)  # normalize values
        flattened.append(state.rpm)
        flattened.append(state.speed)
        # change time to seconds passed
        # flattened.append((state.timestamp - orig_time)/1e12)
        #flattened = scaler.transform(np.asarray(flattened).reshape(1,-1))
        #flattened = np.concatenate((np.asarray(prev_wheel_state).reshape(1, 2), flattened), axis = 1)
        flattened = np.asarray(flattened).reshape(1, -1)
        flattened = np.concatenate((np.asarray(wheel_state).reshape(1, 2), flattened), axis = 1)
        flattened = torch.FloatTensor(flattened)
        return flattened

    def flatten_vector(self, v):
        vector = []
        vector.append(v.x_val)
        vector.append(v.y_val)
        vector.append(v.z_val)
        return vector
