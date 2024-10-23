import os
from MINDMELD.Car import Record as record
import time as sleep
import pickle
import torch
import tkinter as tk
import numpy as np
import math
from matplotlib import pyplot as pl
from MINDMELD.Car.MeLD.ground_truths.rrt_star import RRTStar

base_path='C:/Users/core-robotics/AirSim_Lfd/MINDMELD/Car'
def save_pretest(car,env_num,rollout_num):
    """
    saves initial wizard-of-oz dagger rollout
    Args:
        env_num: int-- environment number
        rollout_num: int-- trajectory to rollout
        car: obj-- Airsim car object

    """

    print("env_num",env_num)
    car.reset(env_num)
    enter = input("Press Enter to start record: ")

    path_save_dir = os.path.join(base_path,'Data', 'MeLD', 'Pretest','Dagger_rollouts','env_'+str(env_num),'roll_out_' + str(rollout_num))

    if not os.path.exists(path_save_dir):
        print('does not exist')
        os.makedirs(os.path.join(path_save_dir))

    print("READY")
    car.step([0, 0])
    sleep.sleep(3)
    print("GO")
    record.record_states_actions(path_save_dir,car)
    print("FINISHED")
    car.step([0, 0])
    car.client.enableApiControl((True))

def get_rrt_path(x_start,y_start,theta_start,dagger_path,save_dir,env,rollout,trial,speed,phsi_orig):
    # Set Initial parameters
    start = [x_start, y_start, theta_start]
    car_length = 4.2

    origin = [56.90, -1.00]
    obstacles = [[70.00, 47.50, 20, 20], [20, 17.5, 20, 20], [80, -22.5, 40, 40], [66.8, 15, 5, 5], [20, -42.5, 20, 20],
                 [89, 500.9, 5, 5], [106.6, 479, 5, 5], [110, 507.6, 5, 5], [127.6, 478.4, 5, 5], [136, 511.7, 5, 5],
                 [150.8, 478.7, 5, 5], [158.1, 511.7, 5, 5], [70.1, 466.3, 10, 10], [70.1, 476.3, 20, 10],
                 [80.1, 516.3, 10, 40], [89, 500.9, 5, 5], [20.1, 496.3, 10, 10], [20.1, 506.3, 20, 10],
                 [32.4, 475.8, 1, 1], [38.2, 476.3, 1, 1], [42.7, 478.4, 1, 1], [46.8, 481.1, 1, 1],
                 [51.8, 484.8, 1, 1], [55.4, 488.2, 1, 1], [57.2, 493.1, 1, 1], [57.9, 497.9, 1, 1]]

    for i in range(len(obstacles)):
        for j in range(len(obstacles)):
            if j == 0:
                obstacles[i][j] -= origin[0]
            if j == 1:
                obstacles[i][j] -= origin[1]


    if env==0:
        x_goal = 34.25
        y_goal = 33.1

        car_length = 4.2

        obstacleList = obstacles[0:4]

        rand_area = [-25, 15, 40.0, 75]
    elif env==1:
        x_goal = 34.25
        y_goal = 33.1

        rand_area = [-45, -5, 45.0, 50]

        car_length = 4.2

        obstacleList = obstacles[0:5]

    elif env==2:

        x_goal = 14.22
        y_goal = 499.22

        rand_area = [10, 480, 120.0, 510]

        car_length = 4.2

        obstacleList = obstacles[5:16]


    elif env==3:

        x_goal = 14.22
        y_goal = 499.22

        rand_area = [-55, 475, 30.0, 515]

        car_length = 4.2

        obstacleList =  obstacles[12:]




    goal = [y_goal, x_goal, 0]

    rrt_star = RRTStar(
        start=start,
        goal=goal,
        rand_area=rand_area,
        obstacle_list=obstacleList,
        car_length = car_length,
        expand_dis=5)
    path = rrt_star.run(save_path = save_dir)
    print("SPEED",speed)

    # Draw final path
    x_val = []
    y_val = []
    print("LEN",len(path))
    for i in range(len(path)):
        x_val.append(path[i][0])
        y_val.append(path[i][1])



    return path, x_goal, y_goal, obstacleList

def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z  # in radians




def get_gt(env_num,rollout_num):
    """
    finds ground truths for wizard-of-oz dagger rollout
    Args:
        env_num: int-- environment number
        rollout_num: int-- trajectory to rollout

    """

    path_save_gt_dir = os.path.join(base_path, 'Data', 'MeLD', 'Pretest','Ground_Truths',  'env_' + str(env_num), 'roll_out_' + str(rollout_num))
    path_pretest = os.path.join(base_path,'Data', 'MeLD', 'Pretest', 'Dagger_rollouts_rollouts', 'env_' + str(env_num),
                                     'roll_out_' + str(rollout_num),'states_actions.pkl')

    if not os.path.exists(path_save_gt_dir):
        os.makedirs(os.path.join(path_save_gt_dir))
    else:
        print("directory already exists")

    with open(path_pretest,'rb') as f:
        data = pickle.load(f)
    pretest_states=data['states']
    time=data['time']


    x_dagger=[]
    y_dagger=[]
    for i in range(0,len(pretest_states)):
        state = pretest_states[i]
        x_dagger.append(state[4])
        y_dagger.append(state[5])

    i=120
    print("env",env_num,"rollout",rollout_num)

    print("LEN",len(pretest_states))
    while i <len(pretest_states):
        print("state ",i)
        state = pretest_states[i]


        state_prev=pretest_states[i-1]
        _, _, yaw_prev = euler_from_quaternion(state_prev[7], state_prev[8], state_prev[9], state_prev[6])

        delta_t=time[i]-time[i-1]


        roll,pitch,yaw=euler_from_quaternion(state[7],state[8],state[9],state[6])
        delta_yaw=yaw-yaw_prev

        phsi_orig=np.arctan(delta_yaw)*5/(state[3]*(delta_t))

        try:
            print("speed", state[3])
            rrt_save_path = os.path.join(path_save_gt_dir, 'rrt_star'+str(i)+'.png')
            rrt_path, x_goal, y_goal, obstacleList = get_rrt_path(y_dagger[i],x_dagger[i],yaw,[y_dagger,x_dagger],rrt_save_path,env_num, rollout_num, i,state[3],phsi_orig)

            with open(os.path.join(path_save_gt_dir,'rrt_path'+str(i)), 'wb') as f:
                pickle.dump(rrt_path, f)

            i+=1
        except Exception:
            break

def smooth_actions():
    base_path= 'C:/Users/core-robotics/AirSim_Lfd/MINDMELD/Car'
    gt_base_path = os.path.join(base_path, 'Data', 'MeLD', 'Pretest', 'Ground_Truths_tuning')
    range_env = {'env_0_rollout_0': [10, 98], 'env_0_rollout_1': [10, 149], 'env_0_rollout_2': [10, 143],
                 'env_0_rollout_3': [10, 132],
                 'env_1_rollout_0': [10, 122], 'env_1_rollout_1': [10, 148], 'env_1_rollout_2': [10, 152],
                 'env_1_rollout_3': [10, 159],
                 'env_2_rollout_0': [10, 153], 'env_2_rollout_1': [10, 165], 'env_2_rollout_2': [10, 116],
                 'env_2_rollout_3': [10, 178],
                 'env_3_rollout_0': [10, 160], 'env_3_rollout_1': [10, 136], 'env_3_rollout_2': [10, 114],
                 'env_3_rollout_3': [10, 121]}

    for e in range(0,4):
        for r in range(0,4):
            print("e",e,"r",r)
            start_stop = range_env['env_'+str(e)+'_rollout_'  + str(r)]
            all_rrt_original=[]
            for i in range(start_stop[0],start_stop[1]-3):
                with open(os.path.join(gt_base_path,
                                       'env_' + str(e), 'roll_out_' + str(r), 'best_action' + str(i)),
                          'rb') as f:
                    rrt = pickle.load(f)
                all_rrt_original.append(rrt)
            all_rrt_original=np.array(all_rrt_original)
            print(all_rrt_original)
            all_rrt = np.zeros(len(all_rrt_original))
            for i in range(len(all_rrt_original)):
                if i < 10:
                    avg = np.mean(all_rrt_original[0:i+10])
                elif i > len(all_rrt_original) - 10:
                    avg = np.mean(all_rrt_original[i-10:-1])
                else:
                    avg = np.mean(all_rrt_original[i-10:i+10])
                all_rrt[i] = avg
            pl.plot(list(range(start_stop[0],start_stop[1]-3)),all_rrt)
            pl.plot(list(range(start_stop[0],start_stop[1]-3)), all_rrt_original)
            pl.savefig(os.path.join(gt_base_path, 'Smoothed', 'plots', 'env_' + str(e) + 'roll_out_' + str(r) + '.png'))
            pl.show()

            save_dir=os.path.join(gt_base_path,'Smoothed',
                                       'env_' + str(e), 'roll_out_' + str(r))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            for i in range(all_rrt.shape[0]):
                with open(os.path.join(save_dir, 'best_action' + str(i+start_stop[0])), 'wb') as f:
                    pickle.dump(all_rrt[i], f)



class Pretest():
    def __init__(self,car,env_num,rollout_num,subj_num=None):
        super(Pretest, self).__init__()
        self.env_num=env_num
        self.rollout_num=rollout_num
        self.car=car
        print("RESETTING CAR")
        self.car.reset(self.env_num)
        self.base_path='C:/Users/core-robotics/AirSim_Lfd/MINDMELD/Car'
        self.subj_num=subj_num
        self.car.client.enableApiControl((True))
        self.total_rollouts=4
        self.total_envs=4



    def rollout(self,env_num=0,rollout_num=0):
        """
        rolls out dagger trajectory
        Args:
            env_num: int-- environment number
            rollout_num: int-- trajectory to rollout

        """
        enter = input("Press Enter to start playback: ")
        sleep.sleep(5)
        print("Playing Rollout")
        print("rollout num", rollout_num)
        path_actions = os.path.join(self.base_path,'Data', 'MeLD', 'Pretest', 'Dagger_rollouts', 'env_' + str(env_num),
                                    'roll_out_' + str(rollout_num),'states_actions.pkl')

        with open(path_actions,
                  'rb') as f:
            data = pickle.load(f)

        actions=data['steering']
        timing=data['time']

        v = 1
        t=0
        total=0
        for a in actions:
            s = sleep.time()
            # assuming throttle is always 1
            if t < len(timing) - v:
                total+=(timing[t + v] - timing[t] )/1000
                while sleep.time() - s < (timing[t + v] - timing[t] ):
                    pass
                t = t + v

        self.car.step([0, 0])
        self.car.client.enableApiControl((True))

    def record_corrective(self,tk_obj,env_num=0,rollout_num=0):
        """
        rolls out dagger trajectory and saves corrective feedback
        Args:
            env_num: int-- environment number
            rollout_num: int-- trajectory to rollout

        """
        path_actions = os.path.join(self.base_path,'Data', 'MeLD', 'Pretest', 'Dagger_rollouts', 'env_' + str(env_num),
                                    'roll_out_' + str(rollout_num),'states_actions.pkl')

        feedback_dir=os.path.join(self.base_path,'Data', 'MeLD', 'Pretest', 'Feedback','subj_'+str(self.subj_num), 'env_' + str(env_num),'roll_out_' + str(rollout_num))


        if not os.path.exists(feedback_dir):
            print('does not exist')
            os.makedirs(os.path.join(feedback_dir))

        #enter = input("Press Enter to start record: ")
        self.car.reset(env_num)
        print("READY")
        text_ready = tk.Label(tk_obj, text="Ready...",font=("Arial", 35),fg='red')
        text_ready.place(x=90, y=150)
        tk_obj.update()
        self.car.step([0, 0])

        self.car.client.enableApiControl((True))
        sleep.sleep(3)
        print("GO")
        text_go = tk.Label(tk_obj, text="Go!",font=("Arial", 35),fg='green')
        text_go.place(x=90, y=220)
        tk_obj.update()


        self.car.client.enableApiControl((False))
        record.record_actions(feedback_dir,self.car)

        with open(path_actions,
                  'rb') as f:
            data = pickle.load(f)

        actions=data['steering']
        timing=data['time']

        v = 1
        t=0
        total=0
        for a in actions:
            s = sleep.time()
            if t < len(timing) - v:
                total+=(timing[t + v] - timing[t] )/1000
                while sleep.time() - s < (timing[t + v] - timing[t] ):
                    pass
                t = t + v
        self.car.client.stopRecording()
        record.stop_record()
        sleep.sleep(.5)
        record.moveRecording(feedback_dir)
        record.combine_corrective(feedback_dir)

        print("FINISHED")
        self.car.step([0, 0])
        self.car.client.enableApiControl((True))

        text_finished = tk.Label(tk_obj,font=("Arial", 25), text="Finished trial "+str(rollout_num+self.total_rollouts*env_num+1)+" of "+str(self.total_rollouts*self.total_envs))
        text_finished.place(x=90, y=350)
        tk_obj.update()




        return text_ready,text_go,text_finished



    def reset(self,env_num,rollout_num):
        """
        rolls out dagger trajectory and saves corrective feedback
        Args:
            env_num: int-- environment number
            dagger_traj_num: int-- trajectory to rollout

        """
        self.env_num=env_num
        self.rollout_num=rollout_num
        self.car.reset(self.env_num)
