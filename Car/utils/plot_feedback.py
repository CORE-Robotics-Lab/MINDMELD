import os
import numpy as np
import pandas as pd
import pickle

from matplotlib import pyplot as pl

base_path = 'C:/Users/core-robotics/AirSim_Lfd/MINDMELD/Car/Data'

subject = '5035'
rollout_num = 12

for rollout in range(rollout_num):
    with open(os.path.join(base_path, 'MeLD', 'Feedback', 'subj_'+subject, 'env_0', 'roll_out_'+str(rollout),'states_actions_rollout.pkl'), 'rb') as f:
        orig_states_actions = pickle.load(f)
    orig_actions = orig_states_actions['steering']
    for i in range(len(orig_actions)):
        if i < 5:
            avg = np.mean(np.array(orig_actions)[0:i + 5])
        elif i > len(orig_actions) - 5:
            avg = np.mean(np.array(orig_actions)[i - 5:-1])
        else:
            avg = np.mean(np.array(orig_actions)[i - 5:i + 5])
        orig_actions[i] = avg

    with open(os.path.join(base_path, 'MeLD', 'Feedback', 'subj_'+subject, 'env_0', 'roll_out_'+str(rollout),'states_actions.pkl'), 'rb') as f:
        orig_feedback = pickle.load(f)
    orig_feedback = orig_feedback['steering']

    with open(os.path.join(base_path, 'MeLD', 'Corrected_Feedback', 'subj_'+subject, 'env_0', 'roll_out_'+str(rollout),'states_actions.pkl'), 'rb') as f:
        mapped_feedback = pickle.load(f)
    mapped_feedback = mapped_feedback['steering']

    pl.plot(orig_actions)
    pl.plot(orig_feedback)
    pl.plot(mapped_feedback)
    pl.legend(["Agent Actions", "Participant Feedback", "Mapped Feedback"])
    pl.ylabel("Action")
    pl.xlabel("Time")
    pl.title(subject+" rollout "+str(rollout))
    pl.show()



