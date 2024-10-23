import airsim

AIRSIM_PATH='C:/Users/core-robotics/Documents/AirSim'
BASE_PATH='C:/Users/core-robotics/AirSim_Lfd/MINDMELD/Car'
OVERWRITE=True

##Domain configs
NUM_ENV = 1
NUM_INITIAL = 1
NUM_CORRECTIVE = 11

ORIGIN = [56.90, -1.00]
CALIBRATION_ENVS= [{"position": airsim.Vector3r(-20, 70, 0.23828665912151337),
                          "orientation": airsim.to_quaternion(0, 0, -.3)},
                         {"position": airsim.Vector3r(-40, 3.088607263634913e-05, 0.23828665912151337),
                          "orientation": airsim.to_quaternion(0, 0, 0)},
                         {"position": airsim.Vector3r(120, 509, 0.23828665912151337),
                          "orientation": airsim.to_quaternion(0, 0, -2.9)},
                         {"position": airsim.Vector3r(-55, 496, 0.23828665912151337),
                          "orientation": airsim.to_quaternion(0, 0, -.2)}, []]
TEST_ENV=[{"position": airsim.Vector3r(-380, -65, 0.23828665912151337),
                          "orientation": airsim.to_quaternion(0, 0, -.3)}]

PRACTICE_ENV=[{"position": airsim.Vector3r(-56.9, -10, 0.23828665912151337),
                          "orientation": airsim.to_quaternion(0, 0, 0)}]

GOALS=[-256.6 - ORIGIN[0], -65.90 - ORIGIN[1]]


NOISE_LEVEL=.2

CROP_SIZE_X=25
CROP_SIZE_Y=256

DAGGER_STEPS=180

#training configs
DEVICE="cuda"
LFD_TRAINING_EPOCHS=5
LFD_LEARNING_RATE=.0001


#flask configs
FLASK_URL="http://127.0.0.1:5000/"
FLASK_URL_DATA= "http://127.0.0.1:5000/data/"

#meld configs
DEMONSTRATION_RANGES={'0_0': [10, 95], '0_1': [10, 112], '0_2': [10, 140],
                     '0_3': [10, 129],
                     '1_0': [10, 119], '1_1': [10, 145], '1_2': [10, 141],
                     '1_3': [10, 156],
                     '2_0': [10, 127], '2_1': [10, 162], '2_2': [10, 113],
                     '2_3': [10, 175],
                     '3_0': [10, 137], '3_1': [10, 133], '3_2': [10, 87],
                     '3_3': [10, 118]}
NUM_CALIB_ENVS=4
NUM_CALIB_ROLL=4
TRAJECTORY_LENGTH=25
MINDMELD_EPOCHS=80
W_DIM=2
SUBJECT_LIST=list(range(1111,1141))+list(range(1142, 1151))+list(range(2000,2017))+list(range(3000,3013))+list(range(4000,4008))
CROP_SIZE_MELD_X=32
CROP_SIZE_MELD_Y=40
W_D_WEIGHT=.01
MM_INPUT_SIZE=1
NUM_LSTM=1
Z_LSTM_DIM=10
