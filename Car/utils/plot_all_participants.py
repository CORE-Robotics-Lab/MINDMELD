import os
import numpy as np
import pandas as pd

from matplotlib import pyplot as pl
import math

base_path = 'C:/Users/core-robotics/AirSim_Lfd/MINDMELD/Car/Data'

subjects = ['3002','3003','3004','3005','3006','3007']
subjects = list(range(5000,5016))+list(range(5017, 5023))+list(range(5024, 5044))
#subjects = ['3003','3004','3005']
#subj_num = input("Enter subject number: ")
distances=np.zeros((len(subjects),12,3))

j=0
for subj_num in subjects:
    subj_num = str(subj_num)
    open_path = os.path.join(base_path,'Post_Processing_Rollouts', subj_num)
    subj_data=pd.read_csv(os.path.join(open_path,'noise_test_mean_distances10.csv'))
    meld=subj_data['meld']
    dagger=subj_data['dagger']
    bc=subj_data['bc']

    distances[j,:,0]=meld
    distances[j,:,1]=dagger
    distances[j,:,2]=bc

    j+=1

numSteps=12
mean_distances=np.mean(distances,axis=0)
std_distances=np.std(distances,axis=0)/math.sqrt(len(subjects))

pl.plot(mean_distances[:,0],color='b')
pl.fill_between(range(numSteps), mean_distances[:,0]-std_distances[:,0], mean_distances[:,0]+std_distances[:,0], alpha = 0.5,color='b')

pl.plot(mean_distances[:,1],color='r')
pl.fill_between(range(numSteps), mean_distances[:,1]-std_distances[:,1], mean_distances[:,1]+std_distances[:,1], alpha = 0.5,color='r')

pl.plot(mean_distances[:,2],color='g')
pl.fill_between(range(numSteps), mean_distances[:,2]-std_distances[:,2], mean_distances[:,2]+std_distances[:,2], alpha = 0.5,color='g')
pl.legend(['meld', 'dagger', 'bc'])
pl.xlabel('Rollout/Policy')
pl.ylabel('Average Distance to Goal')
pl.show()