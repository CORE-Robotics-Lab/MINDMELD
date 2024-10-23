import os
from torch.utils import data
import torch
import pickle
from PIL import Image
import numpy as np

class MyDataset(data.Dataset):
    def __init__(self,paths,transform=None):
        self.actions=[]
        self.throttle=[]
        self.steering=[]
        self.states=[]
        self.im_names=[]
        self.prev_im=[]
        self.prev_state=[]
        self.prev_action=[]
        for path in paths:
            for dir in os.listdir(path):
                for dir2 in os.listdir(os.path.join(path,dir)):
                    print("training on",os.path.join(path,dir,dir2,'states_actions.pkl'))
                    with open(os.path.join(path,dir,dir2,'states_actions.pkl'), "rb") as f:
                        states_actions_data=pickle.load(f, encoding='bytes')
                    self.steering+=states_actions_data['steering']
                    self.throttle+=states_actions_data['throttle']
                    self.states += states_actions_data['states']

                    for i in range(len(states_actions_data['images'])):
                        self.im_names.append(os.path.join(path,dir,dir2,'images',states_actions_data['images'][i]))

        self.transform=transform

    def __getitem__(self, index):
        states=torch.FloatTensor(self.states[index])
        actions=torch.FloatTensor((self.steering[index],self.throttle[index]))
        im_name=self.im_names[index]
        try:
            im = Image.open(im_name)
            im = np.asarray(im)
            if self.transform is not None:
                im = self.transform(im)
            self.prev_im=im
            self.prev_action=actions
            self.prev_state=states
        except Exception:
            im=self.prev_im
            states=self.prev_state
            actions=self.prev_action

        return im,states,actions

    def __len__(self):
        return len(self.steering)
