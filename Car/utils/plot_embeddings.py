from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

path='C:\\Users\\core-robotics\\Downloads\\parsed_LfD_Training_Testing.csv'
data=pd.read_csv(path)
embedding1=data['embedding1']
embedding2=data['embedding2']
amp=data['signed_distance']
delay=data['signed_delay']
velocity=data['velocity']

type=[]
mag=[]
print("LEN",len(amp))
print(embedding1)
for i in range(len(amp)):
    if delay[i]>0 :
        type.append(0)

    elif delay[i]<0 :
        type.append(1)
        print("APPEND LESS THAN 0")
        print(amp[i],delay[i])
    mag.append(abs(delay[i]))
df1 = pd.DataFrame(dict(w_x=embedding1, w_y=embedding2,mag=np.array(mag),type=np.array(type)))
#0: +amp, +delay, 1: +amp, -delay, 2: -amp, +delay, 3: -amp, -delay
colors = {0:'red', 1:'green'}
#plt.scatter(embedding1,embedding2,s=15*df1['mag'],  alpha=0.3,
#            c= df1['type'].map(colors),
#            cmap='viridis',marker='x')

type=[]
mag=[]
print("LEN",len(amp))
print(embedding1)
for i in range(len(amp)):
    if amp[i]>0 :
        type.append(0)

    elif amp[i]<0 :
        type.append(1)
        print("APPEND LESS THAN 0")
        print(amp[i],delay[i])
    mag.append(abs(amp[i]))
df1 = pd.DataFrame(dict(w_x=embedding1, w_y=embedding2,mag=np.array(mag),type=np.array(type)))
#0: +amp, +delay, 1: +amp, -delay, 2: -amp, +delay, 3: -amp, -delay
colors = {0:'blue', 1:'red'}
plt.scatter(embedding1,embedding2,s=500*df1['mag'],  alpha=0.3,
            c= df1['type'].map(colors),
            cmap='viridis',marker='o')

plt.scatter([0.29316895970580537],[1.1139788389205934],marker='x',color='green',s=300)



plt.show()
print(type)