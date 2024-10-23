import torch
import torch.nn as nn
from MINDMELD.Car.Baselines.DAgger.DAgger import Dagger_Agent
import os
import pickle
import numpy as np
import torch.utils.data as utils
import shutil
import time
from MINDMELD.Car.utils.Meld_DataSet import Meld_Dataset
from torchvision import transforms
import sys
from MINDMELD.Car.config import *
from torchvision.transforms.functional import crop


class Meld_Agent(Dagger_Agent):
    def __init__(self, car, env_num, rollout_num, subj_num):
        super().__init__(car, env_num, rollout_num, subj_num=subj_num, agent_name="MeLD")

        self.subj_num = subj_num
        self.car = car
        self.env_num = env_num
        self.rollout_num = rollout_num

        self.device = DEVICE

        self.subjects = SUBJECT_LIST
        self.w_dim = W_DIM


        self.transform = transforms.Compose(
            [transforms.ToPILImage(), transforms.ToTensor(), transforms.Lambda(self.crop800),
             transforms.Resize([CROP_SIZE_MELD_X, CROP_SIZE_MELD_Y])])

    def crop800(self, image):
        return crop(image, 20, 20, 80, 215)

    def __smooth_mapped_actions(self, original_corrective, new_actions):
        original_corrective['steering'] = original_corrective['steering'][:8] + new_actions[8:]

        for i in range(len(original_corrective['steering'])):
            if i < 5:
                avg = np.mean(np.array(original_corrective['steering'])[0:i + 5])
            elif i > len(original_corrective['steering']) - 5:
                avg = np.mean(np.array(original_corrective['steering'])[i - 5:-1])
            else:
                avg = np.mean(np.array(original_corrective['steering'])[i - 5:i + 5])
            original_corrective['steering'][i] = avg
        return original_corrective

    def __save_mapped_action(self, path, original_corrective):
        with open(path, 'wb') as f:
            pickle.dump(original_corrective, f)

        images_source = os.path.join(self.base_path, 'Data', 'MeLD', "Feedback", 'subj_' + str(self.subj_num),
                                     'env_' + str(self.env_num), 'roll_out_' + str(self.rollout_num), 'images')
        image_dest = os.path.join(self.base_path, 'Data', 'MeLD', "Corrected_Feedback", 'subj_' + str(self.subj_num),
                                  'env_' + str(self.env_num), 'roll_out_' + str(self.rollout_num), 'images')

        shutil.copytree(images_source, image_dest)

        path_save_feedback = os.path.join(self.base_path, 'Data', self.agent_name, 'Corrected_Feedback', 'subj_' +
                                          str(self.subj_num), 'env_' + str(self.env_num),
                                          'roll_out_' + str(self.rollout_num), 'states_actions.pkl')

        with open(path_save_feedback, 'wb') as f:
            pickle.dump(original_corrective, f)

    def map(self):
        """
        maps suboptimal labels to improved labels based upon learned personalized embedding. Saves new, better actions in states_actions pickle file

        """

        model = MapModel(num_subj=1, ).to(self.device)

        model.load_state_dict(torch.load(os.path.join(self.base_path, 'Data', 'MeLD', 'Mapper_Models', 'subj_' +
                                                      str(self.subj_num), 'mapper_test.pth')))
        with open(os.path.join(self.base_path, 'Data', 'MeLD', 'Feedback', 'subj_' + str(self.subj_num),
                               'env_' + str(self.env_num), 'roll_out_' + str(self.rollout_num), 'states_actions.pkl'),
                  'rb') as f:
            original_corrective = pickle.load(f)

        eval_dataloader = self.get_dataLoader_test_subj()
        new_actions = self.evaluate_test_subj(dataloader=eval_dataloader, model=model)
        for i in range(len(new_actions)):
            new_actions[i] = new_actions[i].item()

        original_corrective = self.__smooth_mapped_actions(original_corrective, new_actions)

        actions_dir = os.path.join(self.base_path, 'Data', 'MeLD', "Corrected_Feedback", 'subj_' + str(self.subj_num),
                                   'env_' + str(self.env_num), 'roll_out_' + str(self.rollout_num))
        if not os.path.exists(actions_dir):
            os.makedirs(actions_dir)
        else:
            print("directory already exists")
        path = os.path.join(actions_dir, 'corrective_actions.pkl')

        self.__save_mapped_action(path, original_corrective)

    def train_mapper(self, dataloader, model, optimizer, mseloss):
        """
        trains mindmeld mapper
            Parameters:
                        dataloader (utils.Dataloader): dataloader for training mapper
                        model: pytorch model
                        optimizer: pytorch optimizer
                        mseloss: pytorch mseloss

            Returns:
                    model: pytorch model

        """
        model.train()
        batch_size = 32

        mseloss.to(self.device)

        for index, sample in enumerate(dataloader):
            labels, w_ind, diff, gt, l_to_map, states, im = sample
            labels = labels.to(self.device)
            diff = diff.to(self.device)
            states = states.to(self.device)

            try:
                diff_hat, w_hat = model(labels, w_ind, states, im, batch_size=batch_size)
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory, retrying batch', sys.stdout)
                    sys.stdout.flush()
                    for p in model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    diff_hat, w_hat = model(labels, w_ind, states, im, batch_size=batch_size)
                else:
                    raise e

            diff_loss = mseloss(diff_hat.float(), diff.unsqueeze(1).float())
            ws = model.w

            w_loss = mseloss(w_hat, ws[w_ind.long(), :])
            total_loss = diff_loss + W_D_WEIGHT * w_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        return model

    def evaluate_test_subj(self, dataloader, model):
        """
        maps labels for a subject based upon the subjects learned embedding
            Parameters:
                    dataloader (utils.Dataloader): dataloader for training mapper
                    model: pytorch model


            Returns:
                    new_actions (list): list of mapped actions

        """
        model.eval()

        new_actions = []
        for index, sample in enumerate(dataloader):
            labels, w_ind, _, _, l_to_map, states, im = sample

            w_ind = 0
            labels = labels.to(self.device)
            states = states.to(self.device)
            diff_hat, w_hat = model(labels, w_ind, states, im, batch_size=1)
            diff_hat_detached = diff_hat
            diff_hat_detached = diff_hat_detached.detach().to('cpu')
            if diff_hat_detached < -2.5:
                diff_hat_detached = torch.tensor([-2.5])
            elif diff_hat_detached > 2.5:
                diff_hat_detached = torch.tensor([2.5])

            a = (diff_hat_detached + l_to_map.to('cpu'))
            new_actions.append(a)
        return new_actions

    def train_test_mapper(self, train_test="train",  save_dir="Mapper_Models"):
        """
        sets up training for mindmeld
            Parameters:
                    train_test (string): =="train" if training parameters on training participants, =="train_one" if learning embedding for single test participant


        """

        if train_test == "train_one":
            ws = torch.load(os.path.join(self.base_path, 'Data', 'MeLD', 'Mapper_Models', 'all_train', 'w.pth'))
            w_means = torch.mean(ws, dim=0).to(self.device)
            model = MapModel(num_subj=len(self.subjects),
                             backprop_only_w=True).to(self.device)
            model.load_state_dict(torch.load(
                os.path.join(self.base_path, 'Data', 'MeLD', 'Mapper_Models', 'all_train', 'mapper.pth')))
            model.w = torch.nn.Parameter(w_means.unsqueeze(0), requires_grad=True)
            model_save_dir = os.path.join(self.base_path, "Data", "MeLD", save_dir, "subj_" + str(self.subj_num))
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        else:
            model_save_dir = os.path.join(self.base_path, "Data", "MeLD", "Mapper_Models", "all_train3")
            model = MapModel(num_subj=len(self.subjects)).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

        mseloss = torch.nn.MSELoss()

        model_save_name = os.path.join(model_save_dir, "mapper_test.pth")
        w_save_name = os.path.join(model_save_dir, "w.pth")
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        else:
            print("directory already exists")

        if train_test == "train" or train_test == "train_one":
            dataloader = self.get_dataLoader([1, 2, 3], self.total_rollouts, train_test=train_test)
            for e in range(MINDMELD_EPOCHS):
                if e % 10 == 0:
                    print("E", e)
                    print("W: ", model.w)
                model = self.train_mapper(model=model, dataloader=dataloader,
                                          optimizer=optimizer, mseloss=mseloss)
                torch.save(model.state_dict(), model_save_name)
                torch.save(model.w, w_save_name)

    def get_dataLoader(self, train_envs, num_rollouts, train_test="train"):
        """
        converts dataset to dataloader for training
            Parameters:
                    train_envs (list): environments to train on
                    num_rollouts (int): number of rollouts to train on
                    train_test (string): =="train" if training parameters on training participants, =="train_one" if learning embedding for single test participant


            Returns:
                    dataloader: pytorch dataloader

        """

        if train_test == "train":
            subj = self.subjects
        elif train_test == "train_one":
            subj = [self.subj_num]

        np_traj, id, ground_truth, label_to_map, difference, states, image_paths = self.__get_data(train_envs,
                                                                                                 num_rollouts,
                                                                                                 subjects=subj)
        dataLoader = self.__convert_to_tensor_dataset(np_traj, id, ground_truth, label_to_map, difference, states,
                                                    image_paths)
        return dataLoader

    def get_dataLoader_test_subj(self):
        """
        converts dataset to dataloader for evaluating a subjecyt

            Returns:
                    dataloader: pytorch dataloader
                    """


        np_traj, id, label_to_map, states, image_paths = self.__get_data_test()
        dataLoader = self.__convert_to_tensor_dataset(np_traj, id, [], label_to_map, [], states, image_paths,
                                                    batch_size=1, test=True)
        return dataLoader

    def __load_data(self, base_path, subj, e, r):
        with open(os.path.join(base_path, 'subj_' + str(subj),
                               'env_' + str(self.env_num), 'roll_out_' + str(e), 'states_actions.pkl'),
                  'rb') as f:
            subj_label_states = pickle.load(f)

        with open(os.path.join(base_path, 'subj_' + str(subj),
                               'env_' + str(e), 'roll_out_' + str(r),
                               'states_actions_rollout.pkl'), 'rb') as f:
            agent_states_actions = pickle.load(f)

        states = agent_states_actions['states']
        act = agent_states_actions['steering']
        images = agent_states_actions['images']
        return subj_label_states, states, act, images

    def __get_traj_data(self, subj_labels, l):
        if l < int(TRAJECTORY_LENGTH / 2):
            return np.stack((np.transpose(np.hstack((np.zeros((int(TRAJECTORY_LENGTH / 2) - l)),
                                                     np.array(subj_labels[0:l * 2 + 1]),
                                                     np.zeros(
                                                         (int(TRAJECTORY_LENGTH / 2) - l)))))))
        elif l > len(subj_labels) - int(TRAJECTORY_LENGTH / 2) - 1:
            return np.stack((np.transpose(
                np.hstack((np.zeros((int(TRAJECTORY_LENGTH / 2) - (len(subj_labels) - l) + 1)),
                           np.array(
                               subj_labels[2 * (l - len(subj_labels)) + 1:]),
                           np.zeros((int(TRAJECTORY_LENGTH / 2) - (len(subj_labels) - l) + 1)))))))

        else:
            return np.stack((np.transpose(
                np.array(
                    subj_labels[l - int(TRAJECTORY_LENGTH / 2):l + int(TRAJECTORY_LENGTH / 2) + 1]))))

    def __get_data(self, envs, num_rollouts, subjects):
        """
        gets data for all training subjects
            Parameters:
                    envs (list): environments to train on
                    num_rollouts (int): number of rollouts to train on
                    subjects (list): subjects to train on


            Returns:
                    traj (list): list of trajectory segments
                    id (list): subject ids associated with learned embedding
                    ground_truth (list): ground truth labels
                    label_to_map (list): participant provided corrective labels
                    difference (list): difference between participant label and ground truth
                    all_states (list): agent states
                    all_image_paths (list): agent image paths
        """
        base_path = os.path.join(self.base_path, 'Data', 'MeLD', 'Pretest', 'Feedback')
        gt_base_path = os.path.join(self.base_path, 'Data', 'MeLD', 'Pretest', 'Ground_Truths', 'Smoothed')

        traj = []
        id = []
        ground_truth = []
        difference = []
        label_to_map = []
        all_states = []
        all_image_paths = []
        for e in envs:
            for r in range(num_rollouts):
                slice = DEMONSTRATION_RANGES[str(e) + '_' + str(r)]
                subj_id = 0

                rrt_label = []
                prev_rrt = 0
                for i in range(slice[0], slice[1]):

                    with open(os.path.join(gt_base_path,
                                           'env_' + str(e), 'roll_out_' + str(r), 'best_action' + str(i)),
                              'rb') as f:
                        rrt = pickle.load(f)
                        if rrt > 2.5 or rrt < -2.5:
                            rrt = prev_rrt
                        else:
                            prev_rrt = rrt
                    rrt_label.append(rrt)
                for p in subjects:
                    subj_label_states, states, act, images = self.__load_data(base_path, p, e, r)

                    s = []

                    for i in range(slice[0], slice[1]):
                        s.append([states[i][3], act[i]])

                        subj_labels = subj_label_states['steering'][slice[0]:slice[1]]

                    for l in range(0, len(subj_labels)):
                        traj.append(self.__get_traj_data(subj_labels, l))

                        id.append(subj_id)
                        label_to_map.append(subj_labels[l])
                        all_states.append([s[l][0],
                                           subj_labels[l]])
                        difference.append(rrt_label[l] -
                                          subj_labels[l])
                        ground_truth.append(rrt_label[l])
                        all_image_paths.append(os.path.join(base_path, 'subj_' + str(self.subj_num),
                                                            'env_' + str(self.env_num),
                                                            'roll_out_' + str(self.rollout_num),
                                                            "images", images[l]))

                    subj_id += 1

        return traj, id, ground_truth, label_to_map, difference, all_states, all_image_paths

    def __get_data_test(self):
        """
        gets data for single test subject

            Returns:
                    traj (list): list of trajectory segments
                    id (list): subject ids associated with learned embedding
                    ground_truth (list): ground truth labels
                    label_to_map (list): participant provided corrective labels
                    difference (list): difference between participant label and ground truth
                    all_states (list): agent states
            """

        base_path = os.path.join(self.base_path, 'Data', 'MeLD', 'Feedback')
        traj = []
        id = []
        label_to_map = []
        all_states = []
        all_image_paths = []
        time.sleep(4)

        subj_label_states, states, act, images = self.__load_data(base_path, self.subj_num, self.env_num,
                                                                  self.rollout_num)
        subj_labels = subj_label_states['steering']

        s = []
        for i in range(len(states)):
            s.append([states[i][3], act[i]])

        for l in range(0, len(subj_labels)):
            traj.append(self.__get_traj_data(subj_labels, l))
            id.append(0)
            label_to_map.append(subj_labels[l])
            all_states.append([s[l][0],
                               subj_labels[l]])
            all_image_paths.append(os.path.join(base_path, 'subj_' + str(self.subj_num),
                                                'env_' + str(self.env_num), 'roll_out_' + str(self.rollout_num),
                                                "images", images[l]))

        np_traj = np.zeros((len(traj), TRAJECTORY_LENGTH,NUM_LSTM))
        for t in range(len(traj)):
            np_traj[t, :, 0] = traj[t]
        label_to_map = np.array(label_to_map)

        return traj, id, label_to_map, all_states, all_image_paths

    def __convert_to_tensor_dataset(self, np_traj, id, ground_truth, label_to_map, difference, states, images,
                                  batch_size=32, test=False):

        if test:
            dataset = Meld_Dataset(np_traj, id, [], [], label_to_map, states, images, transform=self.transform,
                                   train_test='test')  # create your datset
            dataLoader = utils.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        else:

            dataset = Meld_Dataset(np_traj, id, difference, ground_truth,
                                   label_to_map, states, images, transform=self.transform,
                                   train_test='train')  # create your datset
            dataLoader = utils.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        return dataLoader



class CNN(nn.Module):
    def __init__(self, hidden_size1, hidden_size2, hidden_size3):
        super(CNN, self).__init__()
        # input layer to hidden layer
        self.hidden1 = nn.Linear(768, hidden_size1)
        # hidden layer to output layer
        self.hidden2 = nn.Linear(hidden_size1, hidden_size2)
        self.output = nn.Linear(hidden_size2, hidden_size3)
        self.relu = nn.ReLU()

        self.layer1_cnn = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.layer2_cnn = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.drop_out = nn.Dropout(.2)

    def forward(self, im):
        im = self.layer1_cnn(im.float())
        im = self.layer2_cnn(im)
        im = self.drop_out(im)
        im = im.reshape(im.shape[0], -1)

        out = self.hidden1(im)
        out = self.drop_out(out)
        out = self.hidden2(self.relu(out))
        out = self.output(self.relu(out))

        return out

class Decoder(nn.Module):
    def __init__(self, input_size=14, fc1_size=64, fc2_size=32, out_size=10):
        super().__init__()

        self.input_size = input_size
        self.fc1 = torch.nn.Linear(self.input_size, fc1_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, out_size)

    def forward(self, embed, w):
        z = torch.cat((embed, w), dim=1)
        hidden = self.fc1(z)
        hidden = self.fc2(self.relu(hidden))
        output = self.fc3(self.relu(hidden))

        return output


class Recreate_Model(nn.Module):
    def __init__(self, input_size=14, hidden_size=200, out_size=10):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, out_size)

    def forward(self, z, o_hat):
        x = torch.cat((z, o_hat), dim=1)
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)

        return output




class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=32, fc1_size=128, fc2_size=64, output_size=1, traj_length=10):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True,
                            bidirectional=True)  # batch, seq, feature

        self.fc1 = nn.Linear(hidden_layer_size * 2 * traj_length, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, output_size)
        self.relu = torch.nn.ReLU()

    def forward(self, input_seq, batch_size):
        self.hidden_cell = (torch.zeros(1, batch_size, self.hidden_layer_size).cuda(),
                            torch.zeros(1, batch_size, self.hidden_layer_size).cuda())
        lstm_out, h_out = self.lstm(input_seq)

        lstm_out = lstm_out.view(lstm_out.shape[0], lstm_out.shape[1], 2, self.hidden_layer_size)
        bi_dir = torch.cat((lstm_out[:, :, 0, :], lstm_out[:, :, 1, :]), dim=2)
        bi_dir = torch.flatten(bi_dir, start_dim=1)
        predictions = self.fc1(bi_dir.squeeze())
        predictions = self.fc2(self.relu(predictions))
        z = self.fc3(self.relu(predictions))
        return z


class MapModel(nn.Module):
    def __init__(self,  num_subj=40,
                 state_size=2, backprop_only_w=False):
        super().__init__()
        self.hidden_state_size1 = 64
        self.hidden_state_size2 = 32
        self.z_size1 = 64
        self.z_size2 = 32
        self.recreate = Recreate_Model(input_size=self.z_size2 +MM_INPUT_SIZE, out_size=W_DIM)
        self.lstm_embed = LSTM(input_size=NUM_LSTM, output_size=Z_LSTM_DIM, traj_length=TRAJECTORY_LENGTH)

        self.decode = Decoder(input_size=self.z_size2 +W_DIM, out_size=MM_INPUT_SIZE)
        self.linear_z1 = nn.Linear(state_size + Z_LSTM_DIM, self.z_size1)
        self.linear_z2 = nn.Linear(self.z_size1, self.z_size2)
        self.cnn = CNN(64, 32, 16)

        all_w = torch.zeros((num_subj,W_DIM))
        self.f_state1 = nn.Linear(Z_LSTM_DIM + state_size, self.hidden_state_size1)
        self.f_state2 = nn.Linear(self.hidden_state_size1, self.hidden_state_size2)
        self.relu = torch.nn.ReLU()
        self.tanh = nn.Tanh()
        self.relu = torch.nn.ReLU()

        for i in range(num_subj):
            w = torch.rand(W_DIM)
            all_w[i, :] = w
        self.w = torch.nn.Parameter(all_w)
        if backprop_only_w:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, input_seq, w_ind, s, im, batch_size=32):
        w_set = self.w[w_ind, :]

        if len(w_set.shape) < 2:
            w_set = w_set.unsqueeze(0)

        z_lstm = self.lstm_embed(input_seq, batch_size)

        if batch_size == 1 or len(z_lstm.shape) < 2:
            z_lstm = z_lstm.unsqueeze(0)

        z = torch.cat((z_lstm, s), dim=1)
        z = self.linear_z1(z)
        z = self.linear_z2(self.relu(z))
        o_hat = self.decode(z, w_set)
        w_hat = self.recreate(z, o_hat)
        o_hat = self.tanh(o_hat) * 2.5

        return o_hat, w_hat
