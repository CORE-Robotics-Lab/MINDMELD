from flask import Flask
from flask import request
from flask import render_template
import pickle
import time
import json
import logging
import numpy as np


app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

save_path = 'C:/Users/core-robotics/AirSim_LfdCode/MINDMELD/Car/Records/test_data/'

status = ''
home_wheel = True
replay = False
nn_control = False
nn_value = {'value': 0}
counter = 0
trial = 0
delay_counter = 0
wheel_pos = 0
throttle_pos = 0
test = False
max_vel = .85
dict = {}
wheel_trajectory = []
steering=[]
throttle=[]
time=[]



@app.route('/', methods = ['POST', 'GET'])
def hello():
    global status
    global wheel_trajectory
    global steering
    global throttle
    global time
    global car_trajectory
    global save_path
    global car_env
    global replay
    global counter
    global wheel_vals
    global trial
    global nn_control
    global nn_value
    global home_wheel
    global model
    global steering_record
    global prev_pos
    global prev_prev_pos
    global prev_vel
    global prev_time
    global prev_throttle
    global prev_prev_throttle
    global scaler
    global delay_counter

    global first
    global test

    if request.json is not None:
        content = request.json
        if content['message'] == 'record_actions':
            print('starting record actions')
            #wheel_trajectory = []
            steering=[]
            throttle=[]
            time=[]
            save_path = content['action_path']
            status = "start_record_actions"
        elif content['message'] == 'run':
            status='run'
        elif content['message'] == 'stop_record':
            print("stopping record")
            print("saving", save_path)
            with open(save_path, "wb") as f:
                pickle.dump({'throttle':throttle,'steering':steering,'time':time}, f)
            print("Wheel trajectory recorded! --- with time step = {}".format(len(wheel_trajectory)))
            status = "stop"
            wheel_trajectory = []
            steering = []
            throttle = []
            time = []
        elif content['message'] == 'stop_record_demo':
            print("stopping record")
            status = "stop"
            wheel_trajectory = []
            steering = []
            throttle = []
            time = []
        elif content['message'] == 'home':
            home_wheel=True


    if 'start_record' in request.form:
        print("Use python file to record")
    if 'stop_record' in request.form:
        with open(save_path, "wb") as f:
            pickle.dump({'throttle':throttle,'steering':steering,'time':time}, f)
        print("Wheel trajectory recorded! --- with time step = {}".format(len(wheel_trajectory)))
        status = "stop"
        wheel_trajectory = []
        steering = []
        throttle = []
        time = []


    if "reset" in request.form:
        print("Reset values")
        home_wheel = True
        replay = False
        nn_control = False
        test = False
        nn_value = {'value': 0}
        counter = 0
        trial = 0
        prev_pos = 0
        prev_vel = 0
        prev_time = 0

    if "home_wheel" in request.form or b'home_wheel' in request.data:
        print("Homing wheel")
        home_wheel = True

    if "test" in request.form:
        test = True

    return render_template('base.html')

@app.route('/ff/', methods = ['POST', 'GET'])
def send_ff_value():
    global replay
    global counter
    global nn_control
    global nn_value
    global home_wheel
    global model
    global steering_record
    global prev_pos
    global prev_prev_pos
    global prev_vel
    global prev_time
    global prev_throttle
    global prev_prev_throttle
    global wheel_pos

    if request.method == 'GET':
        if test:
            return {'value': .15}
        if replay:
            counter = counter + 1
            if counter >= len(wheel_vals):
                replay = False
                return {'value': 0}
            else:
                w = wheel_vals[counter]['wheel']-wheel_vals[counter-1]['wheel']
                t = wheel_vals[counter]['time']-wheel_vals[counter-1]['time']
                return {'value': np.tanh(-1.5*w/t)}

        if home_wheel:
            if np.isclose(wheel_pos, 0, .001, .001):
                home_wheel = False
                return {'value': 0}
            elif wheel_pos < -.3:
                return {'value': -.5}
            elif wheel_pos > .3:
                return {'value': .5}
            elif wheel_pos < 0:
                return {'value': -.2}
            elif wheel_pos > 0:
                return {'value': .2}
        else:
            return {'value': 0}
    return render_template('base.html')

@app.route('/data/', methods = ['POST', 'GET'])
def receive_wheel_data():

    global wheel_pos
    global car_state
    global throttle_pos
    global dict

    if request.method == 'POST':
        dict = request.data
        dict = json.loads(request.data)
        wheel_pos=dict['wheel']
        if status == 'start_record_actions':
            time.append(dict['time'])
            steering.append(dict['wheel']*2.5)
            throttle.append(dict['throttle'])
        dict['status']=status
        return dict
    if request.method == 'GET':
        print("status",status)

        return dict
    return "No Data"

