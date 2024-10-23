import requests
import pickle
import time
import os
import shutil
import glob
import pandas as pd
import numpy as np
from MINDMELD.Car.config import *


def record_states_actions(path, car):
    '''
    communicates with flaskapp to record and save states and actions of car.
            Parameters:
                    path (string): path to save states and actions
                    car (Car): car object

    '''

    requests.post(FLASK_URL, json={"message": "run"})


    car.client.enableApiControl(True)
    car.client.startRecording()
    while True:
        response = requests.get(FLASK_URL_DATA)
        wheel_data = response.json()
        if wheel_data['status'] == 'stop':
            print("STOPPED")
            car.client.stopRecording()
            break
        else:
            steering = wheel_data['wheel']
            car.step([steering, 1])
    time.sleep(.5)
    moveRecording(path)


def record_actions(path, car):
    '''
    communicates with flaskapp to record and save actions of car.
            Parameters:
                    path (string): path to save actions
                    car (Car): car object

    '''
    path_actions = os.path.join(path, "corrective_actions.pkl")
    requests.post(FLASK_URL, json={"message": "record_actions", "action_path": path_actions})

    car.client.enableApiControl(True)
    car.client.startRecording()


def stop_record():
    '''
    stops recording
    '''
    requests.post(FLASK_URL, json={"message": "stop_record"})


def combine_corrective(path):
    '''
    downsamples and combines corrective actions from wheel with states from car
            Parameters:
                    path (string): path to states and actions

    '''
    corrective_path = os.path.join(path, 'corrective_actions.pkl')
    time_path = os.path.join(path, 'states_actions.pkl')

    with open(corrective_path,
              'rb') as f:
        corrective = pickle.load(f)
    corrective_time = corrective['time']

    with open(time_path,
              'rb') as f:
        states_actions = pickle.load(f)
    states_actions_time = np.array(states_actions['time'])
    first_time = corrective_time[0]
    corrective_time_new = np.array(np.zeros(len(corrective_time)))
    for i in range(len(corrective_time)):
        corrective_time_new[i] = corrective_time[i]-first_time

    newSteering = []
    newThrottle = []
    for i in range(len(states_actions_time)):
        index = np.argmin(abs(corrective_time_new[:]-states_actions_time[i]))
        newSteering.append(corrective["steering"][index])
        newThrottle.append(corrective["throttle"][index])

    os.rename(os.path.join(path, 'states_actions.pkl'), os.path.join(path, 'states_actions_rollout.pkl'))  # states and actions that were recorded during rollout
    with open(os.path.join(path, 'states_actions.pkl'), "wb") as f:
        pickle.dump({'states': states_actions['states'], "steering": newSteering,
                     "throttle": states_actions['throttle'], 'time': states_actions_time,
                     'images': states_actions['images']}, f)  # replace actions recorded during rollout with corrective feedback



def moveRecording(target_dir):
    '''
    moves recording of states and images from airsim from airsim directory to path
            Parameters:
                    path (string): path to move states and images to

    '''
    recent = max(glob.glob(os.path.join(AIRSIM_PATH, '*/')), key=os.path.getmtime)

    source_dir = os.path.join(AIRSIM_PATH, recent)


    file_names = os.listdir(source_dir)
    for file_name in file_names:
        if os.path.isfile(os.path.join(target_dir, 'states_actions')):
            print("FILE ALREADY EXISTS")
        else:
            if file_name == 'airsim_rec.txt':
                current_df = pd.read_csv(os.path.join(source_dir, file_name), sep='\t')
                states = []
                actions = []
                steering = []
                throttle = []
                times = []
                image_names = []
                first_time = current_df.iloc[0]['TimeStamp']
                for i in range(1, current_df.shape[0] - 1, 1):
                    previous_state = list(current_df.iloc[i - 1][
                                              ['Steering', 'Throttle', 'Brake', 'Speed', 'POS_X', 'POS_Y', 'Q_W', 'Q_X',
                                               'Q_Y', 'Q_Z']])
                    prev_im = list(current_df.iloc[i-1][['ImageFile']])
                    time = (current_df.iloc[i-1]['TimeStamp']-first_time)/1000

                    current_label = list([current_df.iloc[i]['Steering'], current_df.iloc[i]['Throttle']])
                    states.append(previous_state)
                    actions.append(current_label)
                    steering.append(current_df.iloc[i]['Steering'])
                    throttle.append(current_df.iloc[i]['Throttle'])

                    times.append(time)
                    image_names.append(prev_im[0])
                with open(os.path.join(target_dir, 'states_actions.pkl'), "wb") as f:
                    pickle.dump({'states': states, 'steering': steering, 'throttle': throttle, 'time': times,
                                 'images': image_names}, f)
            else:
                print(os.path.isdir(target_dir))
                #if os.path.isdir(target_dir) and OVERWRITE:
                #    shutil.rmtree(target_dir)
                shutil.move(os.path.join(source_dir, file_name), target_dir)


