import os
import pickle
import numpy as np
from MINDMELD.Car.config import *
from MINDMELD.Car.MeLD.meld_main import Meld_Main_TK as meld
from MINDMELD.Car.MeLD.meld_main import Meld_Pretest_TK as meld_pre
from MINDMELD.Car.Baselines.DAgger.dagger_main import Dagger_Main_TK as dagger
from MINDMELD.Car.Baselines.Supervised_BC.bc_main import BC_Main_TK as bc
from MINDMELD.Car.initial_demo import Initial_Demo as initial
from MINDMELD.Car.MeLD.meld_main import pretrain_model

pretrain_model_flag=True
if pretrain_model_flag:
    num_env = 1
    num_initial = 1
    pretrain_model()



##enter subj info
subj_num = input("Enter subject number: ")
order = input("Enter condition order (1=Meld, 2=Dagger, 3=BC): ")
order = order.split(',')
print("order: ",order)

#get corrective feedback on calibratoin tasks
m = meld_pre(NUM_CALIB_ENVS, NUM_CALIB_ROLL, subj_num)

##Initialize noise
noise_path = os.path.join(BASE_PATH, 'Data', 'Noise', 'subj_' + str(subj_num))
if not os.path.isdir(noise_path):
    os.mkdir(noise_path)
for rollout_num in range(NUM_INITIAL+NUM_CORRECTIVE):
    noise_file = os.path.join(noise_path, 'noise_' + str(rollout_num))
    noise_list = np.random.normal(0, NOISE_LEVEL, size=200)
    with open(noise_file, 'wb') as f:
        pickle.dump(noise_list, f)

#collect intitial demonstration
init_demo = initial(NUM_ENV, NUM_CORRECTIVE, NUM_INITIAL, subj_num)

#test mindmeld and baselines
for i in order:
     i = int(i)
     if i == 1:
         m = meld(NUM_ENV, NUM_CORRECTIVE, NUM_INITIAL, subj_num)
     elif i == 2:
         d = dagger(NUM_ENV, NUM_CORRECTIVE, NUM_INITIAL, subj_num)
     else:
         b = bc(NUM_ENV, NUM_CORRECTIVE, NUM_INITIAL, subj_num)
