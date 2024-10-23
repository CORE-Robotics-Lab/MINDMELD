from MINDMELD.MeLD_ToyExperiment.generate_path import get_trial
import pickle
import os
import math
import numpy as np
import torch
import torch.utils.data as utils
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as pl

def create_baseline_trajectories(num_goals,num_trials,train_test='train'):
    all_goals=[]
    goal=[200,40]
    for i in range(num_goals):
        all_goals.append(goal.copy())
        for j in range(num_trials):
            get_trial('trial_'+str(j)+'_goal_'+str(i),goal,train_test)
        if train_test=='train':
            goal[1]=40+(i+1)*50
        else:
            goal[1] = 40 + (i + 1) * 75


    with open(os.path.join('Data',train_test,'goals'), 'wb') as f:
        pickle.dump(all_goals, f)

def find_ground_truth_labels(num_goals,num_trials,train_test='train'):
    with open(os.path.join('Data',train_test, 'goals'), 'rb') as f:
        goals = pickle.load(f)
    for i in range(num_goals):
        for j in range(num_trials):
            print("I",i)
            with open(os.path.join('Data',train_test,'Dagger-rollouts','trial_'+str(j)+'_goal_'+str(i)), 'rb') as f:
                x,y = pickle.load(f)
            x= np.convolve(x, np.ones(5) / 5, mode='valid')
            y = np.convolve(y, np.ones(5) / 5, mode='valid')
            g=goals[i]
            all_difference=[]
            for k in range(len(x)-1):
                ax=g[0]-x[k]
                ay=g[1]-y[k]
                bx=x[k + 1] - x[k]
                by=y[k + 1] - y[k]
                s=ax*by-ay*bx
                c=ax*bx+ay*by
                theta=math.atan2(s,c)
                all_difference.append(theta)

            all_difference=np.convolve(all_difference, np.ones(10) / 10, mode='valid')
            with open(os.path.join('Data',train_test,'ground_truth','trial_'+str(j)+'_goal_'+str(i)+'groundtruthlabels'), 'wb') as f:
                pickle.dump(all_difference, f)
            pl.plot(x[0:-2],y[0:-2])
            pl.plot(g[0],g[1],'x')
            pl.show()
            pl.plot(range(len(all_difference)),all_difference)
            pl.show()



def get_one_subject_data(num_goals,num_trials,type=0,num_steps=3,mag=1.5,ID=1,train_test='train',load_subjects=True):
    if not os.path.exists(os.path.join('Data',train_test,'subj_'+str(ID))):
        os.makedirs(os.path.join('Data',train_test,'subj_'+str(ID)))
    if not load_subjects:
        with open(os.path.join('Data','train','subj_'+str(ID),'style'), 'wb') as f:
            pickle.dump([mag,type], f)

    for i in range(num_goals):
        for j in range(num_trials):
            with open(os.path.join('Data',train_test,'ground_truth','trial_'+str(j)+'_goal_'+str(i)+'groundtruthlabels'), 'rb') as f:
                label = pickle.load(f)
            mapped_label=[]

            if type==0:
                #delay
                for k in range(num_steps):
                    mapped_label.append(label[0]*mag)

                for l in range(len(label)-num_steps):
                    mapped_label.append(label[l]*mag)
            elif type==1:
                #anticipate
                for l in range(num_steps,len(label)):
                    mapped_label.append(label[l] * mag)
                for k in range(num_steps):
                    mapped_label.append(label[-1] * mag)
            elif type==2:
                #none
                for l in range(len(label)):
                    mapped_label.append(label[l] * mag)
            pl.plot(range(len(label)),label)
            pl.plot(range(len(mapped_label)),mapped_label)
            pl.show()
            with open(os.path.join('Data',train_test,'subj_'+str(ID),'trial_'+str(j)+'_goal_'+str(i)), 'wb') as f:
                pickle.dump(mapped_label, f)

def get_all_subject_data(num_subjects,num_goals,num_trials,load_subjects=True,train_test='test'):
    for i in range(num_subjects):
        if load_subjects:
            with open(os.path.join('Data','train','subj_'+str(i),'style'),
                      'rb') as f:
                mag,t = pickle.load(f)
        else:
            mag = np.random.rand() * 2
            t = np.random.randint(low=0, high=3)
            print(mag, t)
        get_one_subject_data(num_goals, num_trials, type=t, mag=mag, ID=i,load_subjects=load_subjects,train_test=train_test)


def create_data_loader(traj_size,batch_size,train_test='test'):
    traj = []
    id = []
    ground_truth = []
    difference=[]
    label_to_map=[]

    for i in range(num_goals):
        for j in range(num_trials):
            with open(os.path.join('Data', train_test,'ground_truth', 'trial_' + str(j) + '_goal_' + str(i) + 'groundtruthlabels'),
                      'rb') as f:
                label = pickle.load(f)
            for p in range(num_subjects):
                with open(os.path.join('Data', train_test,'subj_' + str(p), 'trial_' + str(j) + '_goal_' + str(i)), 'rb') as f:
                    subj_label = pickle.load(f)

                for l in range(0, len(subj_label) - traj_size):
                    traj.append(subj_label[l:l + traj_size])
                    id.append(p)
                    ground_truth.append(label[l + int(traj_size/2)])
                    label_to_map.append(subj_label[l +  int(traj_size/2)])
                    difference.append(label[l +  int(traj_size/2)]-subj_label[l +  int(traj_size/2)])

    np_traj=np.zeros((len(traj),traj_size))
    for t in range(len(traj)):
        np_traj[t,:]=traj[t]
    id=np.array(id)
    ground_truth=np.array(ground_truth)
    label_to_map=np.array(label_to_map)
    difference=np.array(difference)

    if train_test=='train':
        np_traj_train, np_traj_test, id_train, id_test,difference_train,difference_test,ground_truth_train,ground_truth_test,label_to_map_train,label_to_map_test = train_test_split(np_traj, id,difference,ground_truth,label_to_map, test_size = 0.2, random_state = 42)
        np_traj_train=torch.FloatTensor(np_traj_train).unsqueeze(2)
        id_train = torch.FloatTensor(id_train)
        difference_train = torch.FloatTensor(difference_train)
        ground_truth_train = torch.FloatTensor(ground_truth_train)
        label_to_map_train = torch.FloatTensor(label_to_map_train)

        np_traj_test=torch.FloatTensor(np_traj_test).unsqueeze(2)
        id_test = torch.FloatTensor(id_test)
        difference_test = torch.FloatTensor(difference_test)
        ground_truth_test = torch.FloatTensor(ground_truth_test)
        label_to_map_test = torch.FloatTensor(label_to_map_test)


        #for i in range (difference_test.shape[0]):
        #    print(difference[i],id[i],ground_truth[i])

        train_dataset = utils.TensorDataset(np_traj_train, id_train,difference_train,ground_truth_train,label_to_map_train)  # create your datset
        train_loader = utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        test_dataset = utils.TensorDataset(np_traj_test, id_test,difference_test,ground_truth_test,label_to_map_test)  # create your datset
        test_loader = utils.DataLoader(test_dataset, batch_size=1, shuffle=False)

        torch.save(train_loader, os.path.join('Data',train_test,'Data_Loaders','train_loader.pt'))
        torch.save(test_loader,  os.path.join('Data',train_test,'Data_Loaders','test_loader.pt'))
    else:
        np_traj_train=torch.FloatTensor(np_traj).unsqueeze(2)
        id_train = torch.FloatTensor(id)
        difference_train = torch.FloatTensor(difference)
        ground_truth_train = torch.FloatTensor(ground_truth)
        label_to_map_train = torch.FloatTensor(label_to_map)
        train_dataset = utils.TensorDataset(np_traj_train, id_train,difference_train,ground_truth_train,label_to_map_train)  # create your datset
        train_loader = utils.DataLoader(train_dataset, batch_size=1, shuffle=False)

        torch.save(train_loader, os.path.join('Data', train_test, 'Data_Loaders', 'test_loader.pt'))





num_goals=3
num_trials=5
num_subjects=40
traj_size=10
batch_size=128
train_test='test'
#create_baseline_trajectories(num_goals,num_trials,train_test=train_test)
#find_ground_truth_labels(num_goals,num_trials,train_test=train_test)
get_all_subject_data(num_subjects,num_goals,num_trials,train_test=train_test)

#create_data_loader(traj_size,batch_size,train_test=train_test)






#with open(os.path.join('Data', 'ground_truth', 'trial_' + str(0) + '_goal_' + str(0) + 'groundtruthlabels'), 'rb') as f:
#    label = pickle.load(f)

#with open(os.path.join('Data', 'subj_1', 'trial_' + str(0) + '_goal_' + str(0)), 'rb') as f:
#    new_label = pickle.load(f)

"""pl.plot(label,'blue')
pl.plot(new_label,'orange')
pl.legend(['ground truth','participant label'])
pl.show()
"""
