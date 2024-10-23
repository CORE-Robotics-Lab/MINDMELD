from MINDMELD.Car.Model import Model
import time
from MINDMELD.Car import Record as record
import os
import torch
from torch import optim
from torch import nn
from MINDMELD.Car.utils.LfD_DataSet import MyDataset
from torchvision import transforms
from torch.utils import data as Data
import numpy as np
import tkinter as tk
import shutil
import pickle
from MINDMELD.Car.config import *

class Learner_Agent(object):
    def __init__(self, car, env_num, rollout_num, subj_num=None, name="", all_agents=False):
        super(Learner_Agent, self).__init__()
        self.subj_num = subj_num
        self.steps = 200
        self.agent = Model(64, 16, 10)
        self.env_num = env_num
        self.rollout_num = rollout_num
        self.car = car
        self.base_path = BASE_PATH
        self.agent_name = name
        self.total_rollouts = 3
        self.total_envs = 4
        self.save_train_num = -1
        self.device = DEVICE

        self.all_agents = all_agents
        self.agent.to(self.device)

    def setName(self, name):
        self.name = name

    def set_env_rollout(self, env_num=0, rollout_num=0):
        '''
                Parameters:
                        env_num (int): integer indicating the current environment
                        rollout_num (int): integer indicating the current rollout

        '''
        self.env_num = env_num
        self.rollout_num = rollout_num

    def redo_initial(self, env, rollout):
        '''
            Function to allow participant to redo their initial demonstration if they mess it up. Deletes previous
            intial demo.
                Parameters:
                        env (int): integer indicating the environment in which the intial demo should be redone
                        rollout (int): integer indicating the rollout in which the intial demo should be redone
        '''
        path_data = os.path.join(self.base_path, 'Data', self.agent_name, 'Initial', 'subj_' + str(self.subj_num),
                                 'env_' + str(env), 'roll_out_' + str(rollout))
        shutil.rmtree(path_data)
        if self.all_agents:
            path_data = os.path.join(self.base_path, 'Data', "MeLD", 'Initial', 'subj_' + str(self.subj_num),
                                     'env_' + str(env))
            shutil.rmtree(path_data)

            path_data = os.path.join(self.base_path, 'Data', "DAgger", 'Initial', 'subj_' + str(self.subj_num),
                                     'env_' + str(env))
            shutil.rmtree(path_data)

    def rollout_policy(self, noise=True, random_noise=False, n_file='Noise',  model_folder='Models'):
        """
        rolls out BC policy trajectory
             Parameters:
                            noise (bool): indicates if noise should be applied to actions based upon predefined noise
                            random_noise (bool): indicates if random noise should be applied to actions
                            rollout (int): integer indicating the rollout in which the intial demo should be redone
                            n_file (string): specifies path where predefined noise is saved
                            model_folder (string): specifies folder where models to be rolled out are saved

        """

        self.car.client.enableApiControl(True)
        model_name = os.path.join(self.base_path, 'Data', self.agent_name, model_folder, 'subj_' + str(self.subj_num),
                                  'policy_' + str(self.save_train_num) + '.pth')
        print("Using Model ", model_name)
        self.agent.eval()
        self.agent.load_state_dict(torch.load(model_name))

        state, im = self.car.get_state()

        print("STARTING ROLLOUT")
        if not random_noise:
            noise_path = os.path.join(self.base_path, 'Data', 'Noise', 'subj_' + str(self.subj_num))
            noise_file = os.path.join(noise_path, 'noise_' + str(self.rollout_num))
        else:
            noise_file = n_file

        if noise:
            with open(noise_file, 'rb') as f:
                noise_list = pickle.load(f)

        for i in range(self.steps):
            a = self.agent(im.to(device=self.device), state.to(device=self.device))
            if noise:
                if random_noise:
                    n = np.random.normal(0, NOISE_LEVEL)
                else:
                    n = noise_list[i]
                new_a = [a[0] + n, 1]
            else:
                new_a = a

            self.car.step(new_a)
            dist, position = self.car.get_dist()

            if self.car.check_collision() or dist[0] > 150:
                print("Collided or too far", dist)
                break

        self.car.step([0, 0])
        print("finish rollout")
        return dist, position

    def train(self, folders, save_name='Models'):

        """
        Trains neural network with image and state data to predict next action.
        Saves the neural network in the save_name folder
            Parameters:
                        folders (list): specifies folders where data is saved
                        save_name (string): specifies folder where trained model should be saved

        """
        self.save_train_num += 1
        loader_params = {'batch_size': 32, 'shuffle': True, 'num_workers': 0}
        data_path = []
        for folder in folders:
            data_path.append(
                os.path.join(self.base_path, 'Data', self.agent_name, folder, 'subj_' + str(self.subj_num)))
        print("training on", data_path)
        transform = transforms.Compose(
            [transforms.ToPILImage(), transforms.Grayscale(), transforms.CenterCrop([CROP_SIZE_X, CROP_SIZE_Y]), transforms.ToTensor()])
        dataset = MyDataset(data_path, transform=transform)

        dataLoader = Data.DataLoader(dataset, **loader_params, )
        save_dir = []
        model_save_name = []
        if self.all_agents:
            save_dir.append([os.path.join(self.base_path, 'Data', "BC", save_name, 'subj_' + str(self.subj_num))])
            save_dir.append([os.path.join(self.base_path, 'Data', "Dagger", save_name, 'subj_' + str(self.subj_num))])
            save_dir.append([os.path.join(self.base_path, 'Data', "MeLD", save_name, 'subj_' + str(self.subj_num))])
            for dir in save_dir:
                model_save_name.append([os.path.join(dir[0], 'policy_' + str(self.save_train_num) + '.pth')])
                model_save_name2 = os.path.join(dir[0], 'policy_' + str(self.save_train_num) + '.pth')
                if os.path.isfile(model_save_name2) and not OVERWRITE:
                    raise Exception("Model name already exists")
                else:
                    if not os.path.exists(dir[0]):
                        os.makedirs(os.path.join(dir[0]))
                    else:
                        print("directory already exists")
        else:
            save_dir = os.path.join(self.base_path, 'Data', self.agent_name, save_name, 'subj_' + str(self.subj_num))
            model_save_name = os.path.join(save_dir, 'policy_' + str(self.save_train_num) + '.pth')

            if os.path.isfile(model_save_name) and not OVERWRITE:
                raise Exception("Model name already exists")
            else:
                if not os.path.exists(save_dir):
                    os.makedirs(os.path.join(save_dir))
                else:
                    print("directory already exists")

        criterion_S = nn.MSELoss()

        self.agent.to(self.device)
        optimizer = optim.Adam(self.agent.parameters(), lr=LFD_LEARNING_RATE, betas=(0.9, 0.999))
        running_loss = []

        for epoch in range(LFD_TRAINING_EPOCHS):  # loop over the dataset multiple times
            print("Epoch ", epoch)
            for i, data in enumerate(dataLoader):
                im, s, labels = data
                im = im.to(self.device)
                s = s.to(self.device)
                labels = labels.to(self.device).squeeze()

                optimizer.zero_grad()
                outputS, outputT = self.agent(im, s)
                if len(labels.shape) < 2:
                    labels = labels.unsqueeze(0)
                    try:
                        lossS = criterion_S(outputS.reshape(1, ), labels[:, 0].float())
                    except Exception:
                        lossS = criterion_S(outputS.squeeze(), labels[:, 0].float())
                else:
                    lossS = criterion_S(outputS.squeeze(), labels[:, 0].float())

                loss = lossS
                loss.backward()
                optimizer.step()

                running_loss.append(loss.item())
            if epoch % 10 == 9:  # print every 2000 mini-batches
                running_loss = []
        if self.all_agents:
            for save_name in model_save_name:
                torch.save(self.agent.state_dict(), save_name[0])
        else:
            torch.save(self.agent.state_dict(), model_save_name)

    def record_initial(self, tk_obj):
        """
            Records initial demonstration
            Parameters:
                        tk_obj: TK GUI object

        """
        self.car.reset(self.env_num)

        path_save_initial = os.path.join(self.base_path, 'Data', self.agent_name, 'Initial',
                                         'subj_' + str(self.subj_num),
                                         'env_' + str(self.env_num), 'roll_out_' + str(self.rollout_num))
        if not os.path.exists(path_save_initial):
            os.makedirs(os.path.join(path_save_initial))
        else:
            print("directory already exists")
        print("READY")
        text_ready = tk.Label(tk_obj, text="Ready...", font=("Arial", 35), fg='red')
        text_ready.place(x=90, y=450)
        tk_obj.update()
        time.sleep(3)
        print("GO")
        text_go = tk.Label(tk_obj, text="Go!", font=("Arial", 35), fg='green')
        text_go.place(x=90, y=550)
        tk_obj.update()
        record.record_states_actions(path_save_initial, self.car)
        print("FINISHED")
        self.car.step([0, 0])
        self.car.client.enableApiControl(True)
        print("rollout num", self.rollout_num, self.env_num, self.total_rollouts)
        text_finished = tk.Label(tk_obj, font=("Arial", 25), text="Finished demonstration "
                                                                  + str(
            self.rollout_num + self.total_rollouts * self.env_num + 1))
        text_finished.place(x=90, y=650)
        tk_obj.update()
        copy_path = os.path.join(self.base_path, 'Data', "BC", 'Initial',
                                 'subj_' + str(self.subj_num), 'env_' + str(self.env_num))
        if self.all_agents:
            path_save_initial_meld = os.path.join(self.base_path, 'Data', 'MeLD', 'Initial',
                                                  'subj_' + str(self.subj_num))
            path_save_initial_dagger = os.path.join(self.base_path, 'Data', 'DAgger', 'Initial',
                                                    'subj_' + str(self.subj_num))
            """if os.path.isdir(path_save_initial_meld) and OVERWRITE:
                shutil.rmtree(path_save_initial_meld)
            if os.path.isdir(path_save_initial_dagger) and OVERWRITE:
                shutil.rmtree(path_save_initial_dagger)
                """
            shutil.copytree(copy_path, os.path.join(path_save_initial_meld, 'env_0'))
            shutil.copytree(copy_path, os.path.join(path_save_initial_dagger, 'env_0'))

        return text_ready, text_go, text_finished
