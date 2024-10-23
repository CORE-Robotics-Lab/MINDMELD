"""
Path tracking simulation with Stanley steering control and PID speed control.
author: Atsushi Sakai (@Atsushi_twi)
Ref:
    - [Stanley: The robot that won the DARPA grand challenge](http://isl.ecst.csuchico.edu/DOCS/darpa2005/DARPA%202005%20Stanley.pdf)
    - [Autonomous Automobile Path Tracking](https://www.ri.cmu.edu/pub_files/2009/2/Automatic_Steering_Methods_for_Autonomous_Automobile_Path_Tracking.pdf)
"""
import numpy as np
import matplotlib.pyplot as plt
from rrt_star import RRTStar
import sys
# sys.path.append("../../PathPlanning/CubicSpline/")
#
# try:
#     import cubic_spline_planner
# except:
#     raise


k = 0.5  # control gain
Kp = 1.0  # speed proportional gain
dt = 0.2  # [s] time difference
L = 2.35  # [m] Wheel base of vehicle
max_steer = np.radians(40.0)  # [rad] max steering angle
max_steer_change = np.radians(4.0) # max delta in .2 seconds

show_animation = True


class State(object):
    """
    Class representing the state of a vehicle.
    :param x: (float) x-coordinate
    :param y: (float) y-coordinate
    :param yaw: (float) yaw angle
    :param v: (float) speed
    """

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0, delta = 0.0, prev_delta = 0.0):
        """Instantiate the object."""
        super(State, self).__init__()
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.delta = delta
        self.prev_delta = prev_delta

    def update(self, acceleration, delta):
        """
        Update the state of the vehicle.
        Stanley Control uses bicycle model.
        :param acceleration: (float) Acceleration
        :param delta: (float) Steering
        """
        delta = np.clip(delta, self.prev_delta - max_steer_change, self.prev_delta + max_steer_change)
        delta = np.clip(delta, -max_steer, max_steer)

        self.x += self.v * np.cos(self.yaw) * dt
        self.y += self.v * np.sin(self.yaw) * dt
        self.yaw += self.v / L * np.tan(delta) * dt
        self.delta = delta
        self.yaw = normalize_angle(self.yaw)
        self.v += acceleration * dt
        self.prev_delta = delta


def pid_control(target, current):
    """
    Proportional control for the speed.
    :param target: (float)
    :param current: (float)
    :return: (float)
    """
    return Kp * (target - current)


def stanley_control(state, cx, cy, cyaw, last_target_idx):
    """
    Stanley steering control.
    :param state: (State object)
    :param cx: ([float])
    :param cy: ([float])
    :param cyaw: ([float])
    :param last_target_idx: (int)
    :return: (float, int)
    """
    current_target_idx, error_front_axle, reached_goal = calc_target_index(state, cx, cy)

    if last_target_idx >= current_target_idx:
        current_target_idx = last_target_idx

    # theta_e corrects the heading error
    theta_e = normalize_angle(cyaw[current_target_idx] - state.yaw)
    # theta_d corrects the cross track error
    theta_d = np.arctan2(k * error_front_axle, state.v)
    # Steering control
    delta = theta_e + theta_d

    return delta, current_target_idx, reached_goal


def normalize_angle(angle):
    """
    Normalize an angle to [-pi, pi].
    :param angle: (float)
    :return: (float) Angle in radian in [-pi, pi]
    """
    while angle > np.pi:
        angle -= 2.0 * np.pi

    while angle < -np.pi:
        angle += 2.0 * np.pi

    return angle


def calc_target_index(state, cx, cy):
    """
    Compute index in the trajectory list of the target.
    :param state: (State object)
    :param cx: [float]
    :param cy: [float]
    :return: (int, float)
    """
    # Calc front axle position
    fx = state.x + L * np.cos(state.yaw)
    fy = state.y + L * np.sin(state.yaw)

    # Search nearest point index
    dx = [fx - icx for icx in cx]
    dy = [fy - icy for icy in cy]
    d = np.hypot(dx, dy)
    target_idx = np.argmin(d)

    # Project RMS error onto front axle vector
    front_axle_vec = [-np.cos(state.yaw + np.pi / 2),
                      -np.sin(state.yaw + np.pi / 2)]
    error_front_axle = np.dot([dx[target_idx], dy[target_idx]], front_axle_vec)

    reached_goal = False

    if d[-1] < 5:
        print("got close to goal")
        reached_goal = True

    return target_idx, error_front_axle, reached_goal

def run(start, startyaw, goal, path, speed, init_delta, obstacleList, dagger_path, save_path):
    startx = start[0]
    starty = start[1]
    goalx = goal[0]
    goaly = goal[1]
    cx = [x for (x, y) in path]
    cx.reverse()
    cy = [y for (x, y) in path]
    cy.reverse()

    accel = 3

    # if goaly > starty:
    #     cyaw = [np.pi / 2 - np.arctan((goalx - startx) / (goaly - starty))]
    # else:
    #     cyaw = [np.pi / 2 - np.arctan((goalx - startx) / (goaly - starty)) + np.pi]
    #cyaw = [np.pi/2-startyaw]
    cyaw = []
    step_size = 1
    for i in range(step_size, len(cx), step_size):
        if (cy[i] - cy[i-step_size]) == 0:
            cyaw.append(np.pi)
        else:
            if cy[i] > cy[i-step_size]:
                for j in range(step_size):
                    cyaw.append(np.pi / 2 - np.arctan((cx[i] - cx[i - step_size]) / (cy[i] - cy[i - step_size])))
            else:
                for j in range(step_size):
                    cyaw.append(np.pi / 2 - np.arctan((cx[i] - cx[i - step_size]) / (cy[i] - cy[i - step_size])) + np.pi)
    if len(cx) > len(cyaw):
        for j in range(len(cx)-len(cyaw)):
            cyaw.append(cyaw[-1])

    #print(len(path))

    # print("cx", cx)
    # print("cy", cy)
    # print("cyaw", np.degrees(np.asarray(cyaw)))

    # if goaly > starty:
    #     cyaw =[np.pi / 2 - np.arctan((goalx - startx) / (goaly - starty))]*len(path)
    # else:
    #     cyaw = [np.pi / 2 - np.arctan((goalx - startx) / (goaly - starty)) + np.pi]*len(path)

    # print("angle to goal (for stanley 0 ->)", np.degrees(cyaw[0]))

    target_speed = speed # [m/s]

    max_simulation_time = 50.0

    # Initial state
    print("start yaw", np.degrees(np.pi/2-startyaw))
    state = State(x=startx, y=starty, yaw=np.pi/2-startyaw, v=speed, prev_delta = init_delta)

    print("init delta", init_delta)

    last_idx = len(cx) - 1
    time = 0.0
    x = [state.x]
    y = [state.y]
    yaw = [state.yaw]
    v = [state.v]
    t = [0.0]
    delta_yaw = [0.0]

    target_idx, _, _ = calc_target_index(state, cx, cy)

    while max_simulation_time >= time and last_idx >= target_idx:

        #ai = pid_control(target_speed, state.v)
        ai = accel

        # print("curr x y", state.x, state.y)
        # print("curr yaw", np.degrees(state.yaw))
        # print("goal yaw", np.degrees(cyaw[target_idx]))

        di, target_idx, reached_goal = stanley_control(state, cx, cy, cyaw, target_idx)
        if reached_goal:
            #print(time)
            break
        state.update(ai, di)
        #print("before", state.v)

        time += dt
        #state.v = state.v + accel*dt
        #print("speed", state.v)

        x.append(state.x)
        y.append(state.y)
        yaw.append(state.yaw)
        delta_yaw.append(-state.delta)
        v.append(state.v)
        t.append(time)



        # print("delta", np.degrees(-state.delta))

    # print(np.degrees(np.asarray(yaw)))
    # print(np.degrees(np.asarray(delta_yaw)))

    if show_animation:  # pragma: no cover
        plt.figure()
        plt.plot(startx, starty, "xr", label="start", zorder = 4)
        plt.plot(cx, cy, ".r", label="rrt*", zorder = 1)
        plt.plot(x, y, "-g", label="stanley", zorder = 3)
        plt.plot(dagger_path[1],dagger_path[0], "b", label='woz rollout', zorder = 2)

        plt.plot([], [], marker='>', color='C0', label = "yaw")
        plt.plot([], [], marker='>', color='navy', label = "initial steering angle")
        plt.plot([], [], marker='>', color='darkgreen', label="action")
        plt.legend()

        # plt.arrow(x[0], y[0], 7 * np.sin(0), 7 * np.cos(0), color='black', head_width=1)
        # print("orientation arrow", np.degrees(np.pi/2-yaw[0]))
        # print("action arrow", np.degrees(delta_yaw[1] + np.pi / 2 - yaw[0]))
        # print("best action", np.degrees(delta_yaw[1]))
        plt.arrow(x[0], y[0], 7 * np.sin(init_delta + np.pi / 2 - yaw[0]), 7 * np.cos(init_delta + np.pi / 2 - yaw[0]), color='navy', head_width=1, zorder=6)
        plt.arrow(x[0],y[0], 7*np.sin(delta_yaw[1]+np.pi/2-yaw[0]), 7*np.cos(delta_yaw[1]+np.pi/2-yaw[0]), color='darkgreen', head_width = 1, zorder = 6)
        plt.arrow(x[0],y[0], 7*np.sin(np.pi/2-yaw[0]), 7*np.cos(np.pi/2-yaw[0]), color='C0', head_width = 1, zorder = 5)




        for i in obstacleList:
            plt.plot([i[1], i[1] + i[3]], [i[0], i[0]], color='red')
            plt.plot([i[1], i[1]], [i[0], i[0] + i[2]], color='red')
            plt.plot([i[1] + i[3], i[1] + i[3]], [i[0], i[0] + i[2]], color='red')
            plt.plot([i[1], i[1] + i[3]], [i[0] + i[2], i[0] + i[2]], color='red')

        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.axis("equal")
        plt.grid(True)
        plt.pause(.01)
        plt.savefig(save_path)


        # plt.subplots(1)
        # plt.plot(t, v, "-r")
        # plt.xlabel("Time[s]")
        # plt.ylabel("Speed[m/s]")
        # plt.grid(True)

        plt.close()
        #plt.show()

    ret_yaw = []
    for i in yaw:
        ret_yaw.append(np.pi/2-i)

    return x, y, ret_yaw, v, t, delta_yaw




# def main():
#     """Plot an example of Stanley steering control on a cubic spline."""
#     #  target course
#     # ax = [0.0, 100.0, 100.0, 50.0, 60.0]
#     # ay = [0.0, 0.0, -30.0, -20.0, 0.0]
#
#     # cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
#     #     ax, ay, ds=0.1)
#
#     # env0
#     x_goal = 34.25
#     y_goal = 33.1
#     scale=5
#     xshift=40
#     yshift=40
#
#     obstacleList = [
#         [22.5, -18.5, 10, 35],
#         [-38.05, -39.75, 16, 24],
#         [-31.9, 23.5, 20, 26],
#         [12.01, 47.24, 10, 21]
#     ]  # [x,y,size(radius)]
#
#     startx = 70
#     starty = -20
#     startyaw = np.radians(120)
#
#     # Set Initial parameters
#     rrt_star = RRTStar(
#         start=[70, -20],
#         goal=[33.1, 34.25],
#         rand_area=[-25, 15, 40.0, 75],
#         obstacle_list=obstacleList,
#         expand_dis=1)
#     path = rrt_star.run()
#
#     cx = [x for (x, y) in path]
#     cx.reverse()
#     cy = [y for (x, y) in path]
#     cy.reverse()
#     cyaw = [np.arctan((y_goal - starty) / (x_goal - startx)) + np.pi] * len(path)
#
#     #cx = np.linspace(startx, x_goal, 50)
#     #cy = np.linspace(starty, y_goal, 50)
#     #cx = [40, x_goal]
#     #cy = [10, y_goal]
#     #cyaw = [np.arctan((y_goal-starty)/(x_goal-startx))+np.pi]*2
#
#
#
#     target_speed = 30.0 / 3.6  # [m/s]
#
#     max_simulation_time = 100.0
#
#     # Initial state
#     state = State(x=startx, y=starty, yaw=startyaw, v=30.0)
#
#     last_idx = len(cx) - 1
#     time = 0.0
#     x = [state.x]
#     y = [state.y]
#     yaw = [state.yaw]
#     v = [state.v]
#     t = [0.0]
#     target_idx, _, _ = calc_target_index(state, cx, cy)
#
#     while max_simulation_time >= time and last_idx >= target_idx:
#         ai = pid_control(target_speed, state.v)
#         di, target_idx, reached_goal = stanley_control(state, cx, cy, cyaw, target_idx)
#         if reached_goal:
#             break
#         state.update(ai, di)
#
#         time += dt
#
#         x.append(state.x)
#         y.append(state.y)
#         yaw.append(state.yaw)
#         v.append(state.v)
#         t.append(time)
#
#         # if show_animation:  # pragma: no cover
#         #     plt.cla()
#         #     # for stopping simulation with the esc key.
#         #     plt.gcf().canvas.mpl_connect('key_release_event',
#         #             lambda event: [exit(0) if event.key == 'escape' else None])
#         #     plt.plot(cx, cy, ".r", label="course")
#         #     plt.plot(x, y, "-b", label="trajectory")
#         #     plt.plot(cx[target_idx], cy[target_idx], "xg", label="target")
#         #     plt.axis("equal")
#         #     plt.grid(True)
#         #     plt.title("Speed[km/h]:" + str(state.v * 3.6)[:4])
#         #     plt.pause(0.001)
#
#
#     # Test
#     #assert last_idx >= target_idx, "Cannot reach goal"
#
#     if show_animation:  # pragma: no cover
#         plt.plot(startx, starty, "xr", label="start")
#         plt.plot(cx, cy, ".r", label="course")
#         plt.plot(x, y, "-b", label="trajectory")
#         plt.legend()
#
#         for i in obstacleList:
#             plt.plot([i[1], i[1] + i[3]], [i[0], i[0]], color='red')
#             plt.plot([i[1], i[1]], [i[0], i[0] + i[2]], color='red')
#             plt.plot([i[1] + i[3], i[1] + i[3]], [i[0], i[0] + i[2]], color='red')
#             plt.plot([i[1], i[1] + i[3]], [i[0] + i[2], i[0] + i[2]], color='red')
#
#         plt.xlabel("x[m]")
#         plt.ylabel("y[m]")
#         plt.axis("equal")
#         plt.grid(True)
#
#         plt.subplots(1)
#         plt.plot(t, [iv * 3.6 for iv in v], "-r")
#         plt.xlabel("Time[s]")
#         plt.ylabel("Speed[km/h]")
#         plt.grid(True)
#         plt.show()

#
# if __name__ == '__main__':
#     main()