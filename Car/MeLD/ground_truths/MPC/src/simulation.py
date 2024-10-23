from map import Map, Obstacle
import numpy as np
from reference_path import ReferencePath
from spatial_bicycle_models import BicycleModel
import matplotlib.pyplot as plt
from MPC import MPC
from scipy import sparse
import time
import pickle
import os
import math

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

if __name__ == '__main__':

    base_path = 'C:/Users/core-robotics/AirSim_Lfd/MINDMELD/Car'
    num_env=4
    num_rollouts=4
    start_ends={'env_0_rollout_0':[10,98],'env_0_rollout_1':[10,149],'env_0_rollout_2':[10,143],'env_0_rollout_3':[10,132],
                'env_1_rollout_0':[10,122],'env_1_rollout_1':[10,148],'env_1_rollout_2':[10,152],'env_1_rollout_3':[10,159],
                'env_2_rollout_0':[10,153],'env_2_rollout_1':[10,165],'env_2_rollout_2':[10,116],'env_2_rollout_3':[10,178],
                'env_3_rollout_0':[10,160],'env_3_rollout_1':[10,136],'env_3_rollout_2':[10,114],'env_3_rollout_3':[10,121]}
    for e in range(num_env):
        for r in range(num_rollouts):

            path_save_gt_dir = os.path.join(base_path, 'Data', 'MeLD', 'Pretest','Ground_Truths_tuning',  'env_' + str(e), 'roll_out_' + str(r))
            path_pretest = os.path.join(base_path,'Data', 'MeLD', 'Pretest', 'Dagger_rollouts_rollouts', 'env_' + str(e),
                                             'roll_out_' + str(r),'states_actions.pkl')

            if not os.path.exists(path_save_gt_dir):
                os.makedirs(os.path.join(path_save_gt_dir))
            else:
                print("directory already exists")

            with open(path_pretest,'rb') as f:
                data = pickle.load(f)
            pretest_states=data['states']
            time=data['time']
            steering = data['steering']

            x_dagger=[]
            y_dagger=[]
            for l in range(0,len(pretest_states)):
                state = pretest_states[l]
                x_dagger.append(state[4])
                y_dagger.append(state[5])


            print("LEN",len(pretest_states))
            start=start_ends['env_'+str(e)+'_rollout_'+str(r)][0]
            stop=start_ends['env_'+str(e)+'_rollout_'+str(r)][1]
            i=start

            while i <stop-3:
                print("env ",e," rollout ",r,"step ",i)
                print("state ",i)
                state = pretest_states[i]

                state_prev=pretest_states[i-1]
                delta_t=time[i]-time[i-1]


                roll,pitch,yaw=euler_from_quaternion(state[7],state[8],state[9],state[6])
                yaw=math.pi/2-yaw

                print("YAW", yaw)
                print("YAW degrees", np.degrees(yaw))


                # Load map file
                map = Map(file_path='maps/env_1.png', origin=[ -600,-600],
                          resolution=1)

                ## env0 and env1 are -70 to 90

                # Specify waypoints
                with open(os.path.join('C:\\Users\\core-robotics\\Airsim_Lfd\\MINDMELD\\Car\\Data\\MeLD\\Pretest\\Ground_Truths_tuning','env_'+str(e),'roll_out_'+str(r),'rrt_path'+str(i)),'rb') as f:
                    waypoints_reversed=pickle.load(f)
                print("path length",len(waypoints_reversed))
                print("loading",os.path.join('C:\\Users\\core-robotics\\Airsim_Lfd\\MINDMELD\\Car\\Data\\MeLD\\Pretest\\Ground_Truths_tuning','env_'+str(e),'roll_out_'+str(r),'rrt_path'+str(i)))

                #waypoints=[[499.22, 14.22], [499.66159577486667, 19.009552787782624], [499.70364172651676, 19.46558365674079], [499.38498449951487, 24.303814090487613], [499.2637033709145, 28.448634970600295], [499.3621137963891, 32.71258253293225], [499.34532795875447, 36.00095970754201]]
                wp_x=[]
                wp_y=[]
                waypoints=list(reversed(waypoints_reversed))
                if len(waypoints)>100:
                    sample_rate=20
                else:
                    sample_rate=1

                for j in range(0,len(waypoints),sample_rate):
                    wp_x.append(waypoints[j][0])
                    wp_y.append(waypoints[j][1])

                print("X",wp_x)
                print("Y",wp_y)


                # Specify path resolution
                path_resolution = 1  # m / wp

                # Create reference path
                reference_path = ReferencePath(map, wp_x, wp_y, path_resolution,
                             smoothing_distance=1, max_width=5.0,
                                               circular=False)

                # Instantiate motion model
                car = BicycleModel(length=4.2, width=1.8,
                                   reference_path=reference_path, Ts=.2)
                car.temporal_state.psi=yaw

                # Real-World Environment. Track used for testing the algorithm on a 1:12
                # RC car.


                ##############
                # Controller #
                ##############
                if len(waypoints_reversed)>31:
                    N = 30
                else:
                    N=len(waypoints_reversed)-1

                Q = sparse.diags([1.0, 0.0, 0.0])
                R = sparse.diags([0.5, 50000])
                QN = sparse.diags([1.0, 0.0, 0.0])

                v_max = 25  # m/s
                delta_max = np.radians(40)  # rad
                ay_max = 4.0  # m/s^2
                InputConstraints = {'umin': np.array([0.0, -np.tan(delta_max)/car.length]),
                                    'umax': np.array([v_max, np.tan(delta_max)/car.length])}
                StateConstraints = {'xmin': np.array([-np.inf, -np.inf, -np.inf]),
                                    'xmax': np.array([np.inf, np.inf, np.inf])}
                mpc = MPC(car, N, Q, R, QN, StateConstraints, InputConstraints, ay_max)

                # Compute speed profile
                a_min = 0  # m/s^2
                a_max = 3.0  # m/s^2
                SpeedProfileConstraints = {'a_min': a_min, 'a_max': a_max,
                                           'v_min': 0, 'v_max': v_max, 'ay_max': ay_max}
                print("initial velocity",state[3])
                car.reference_path.compute_speed_profile(SpeedProfileConstraints,init_velocity=state[3])


                ##############
                # Simulation #
                ##############

                # Set simulation time to zero
                t = 0.0

                # Logging containers

                x_log = [car.temporal_state.x]
                y_log = [car.temporal_state.y]
                v_log = [0.0]
                print("starting simulation")

                # Until arrival at end of path
                try:
                    k=0
                    while k < 1:
                        print("temporal state",car.temporal_state.x,car.temporal_state.y,car.temporal_state.psi,yaw)
                        # Get control signals
                        u = mpc.get_control()
                        print("actions",u)

                        action=u[-1]/(np.radians(40))*2.5*-1
                        print("action",action)
                        #
                        prev_action=action
                        with open(os.path.join(path_save_gt_dir,'best_action'+str(i)), 'wb') as f:
                             pickle.dump(action, f)

                        # Simulate car
                        car.drive(u)

                        # Log car state
                        x_log.append(car.temporal_state.x)
                        y_log.append(car.temporal_state.y)
                        v_log.append(u[0])


                        # Increment simulation time
                        t += car.Ts

                        # Plot path and drivable area
                        reference_path.show()

                        # Plot car
                        car.show()

                        # Plot MPC prediction
                        mpc.show_prediction()

                        # Set figure title
                        # plt.title('MPC Simulation: v(t): {:.2f}, delta(t): {:.2f}, Duration: '
                        #          '{:.2f} s'.format(u[0], u[1], t))
                        # plt.axis('off')
                        # plt.pause(2)
                        k+=1
                except Exception:
                    print("DID Not find action")
                    with open(os.path.join(path_save_gt_dir, 'best_action' + str(i)), 'wb') as f:
                        pickle.dump(prev_action, f)
                i+=1
                print("HHHH",e,r)

