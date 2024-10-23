"""
Path planning Sample Code with RRT and Dubins path

author: AtsushiSakai(@Atsushi_twi)

"""

import copy
import math
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../DubinsPath/")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../RRTStar/")

try:
    import MINDMELD.Car.MeLD.ground_truths.rrtDubin.dubins_path_planning as dubins_path_planning
    from MINDMELD.Car.MeLD.ground_truths.rrtDubin.rrt_star import RRTStar
except ImportError:
    raise

show_animation = True


class RRTStarDubins(RRTStar):
    """
    Class for RRT star planning with Dubins path
    """

    class Node(RRTStar.Node):
        """
        RRT Node
        """

        def __init__(self, x, y, yaw):
            super().__init__(x, y)
            self.yaw = yaw
            self.path_yaw = []

    def __init__(self, start, goal, obstacle_list, rand_area_x,rand_area_y,
                 goal_sample_rate=40,
                 max_iter=500,
                 connect_circle_dist=50.0
                 ):
        """
        Setting Parameter

        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacleList:obstacle Positions [[x,y,size],...]
        randArea:Random Sampling Area [min,max]

        """

        self.start = self.Node(start[0], start[1], start[2])
        self.end = self.Node(goal[0], goal[1], goal[2])
        self.min_rand_x = rand_area_x[0]
        self.max_rand_x = rand_area_x[1]
        self.min_rand_y = rand_area_y[0]
        self.max_rand_y = rand_area_y[1]
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.obstacle_list = obstacle_list
        self.connect_circle_dist = connect_circle_dist

        self.curvature = .75  # for dubins path
        self.goal_yaw_th = 100
        self.goal_xy_th = 5

    def planning(self, animation=True, search_until_max_iter=True):
        """
        RRT Star planning

        animation: flag for animation on or off
        """

        self.node_list = [self.start]
        for i in range(self.max_iter):
            print("Iter:", i, ", number of nodes:", len(self.node_list))
            rnd = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd)
            new_node = self.steer(self.node_list[nearest_ind], rnd)

            if self.check_collision(new_node, self.obstacle_list):
                near_indexes = self.find_near_nodes(new_node)
                new_node = self.choose_parent(new_node, near_indexes)
                if new_node:
                    self.node_list.append(new_node)
                    self.rewire(new_node, near_indexes)

            if animation and i % 5 == 0:
                #self.plot_start_goal_arrow()
                #self.draw_graph(rnd)
                pass

            if (not search_until_max_iter) and new_node:  # check reaching the goal
                last_index = self.search_best_goal_node()
                if last_index:
                    return self.generate_final_course(last_index)

        print("reached max iteration")

        last_index = self.search_best_goal_node()
        if last_index:
            return self.generate_final_course(last_index)
        else:
            print("Cannot find path")

        return None

    def draw_graph(self, rnd=None,x_val=None,y_val=None):
        plt.clf()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect('key_release_event',
                                     lambda event: [exit(0) if event.key == 'escape' else None])
        if rnd is not None:
            plt.plot(rnd.x, rnd.y, "^k")
        for node in self.node_list:
            if node.parent:
                plt.plot(node.path_x, node.path_y, "-g")

        for (x, y, w, h) in self.obstacle_list:
            # self.plot_circle(ox, oy, size)
            plt.gca().add_patch(Rectangle((y, x), h, w, linewidth=1, edgecolor='r', facecolor='none'))

        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.end.x, self.end.y, "xr")
        if not x_val is None:
            plt.plot(x_val, y_val,'r')


        plt.axis([-50, 100, -50, 50])
        plt.grid(True)
        self.plot_start_goal_arrow()
        plt.pause(0.01)
        #time.sleep(.2)


    def plot_start_goal_arrow(self):
        dubins_path_planning.plot_arrow(
            self.start.x, self.start.y, self.start.yaw)
        dubins_path_planning.plot_arrow(
            self.end.x, self.end.y, self.end.yaw)

    def steer(self, from_node, to_node):

        px, py, pyaw, mode, course_length = dubins_path_planning.dubins_path_planning(
            from_node.x, from_node.y, from_node.yaw,
            to_node.x, to_node.y, to_node.yaw, self.curvature)

        if len(px) <= 1:  # cannot find a dubins path
            return None

        new_node = copy.deepcopy(from_node)
        new_node.x = px[-1]
        new_node.y = py[-1]
        new_node.yaw = pyaw[-1]

        new_node.path_x = px
        new_node.path_y = py
        new_node.path_yaw = pyaw
        new_node.cost += course_length
        new_node.parent = from_node

        return new_node

    def calc_new_cost(self, from_node, to_node):

        _, _, _, _, course_length = dubins_path_planning.dubins_path_planning(
            from_node.x, from_node.y, from_node.yaw,
            to_node.x, to_node.y, to_node.yaw, self.curvature)

        return from_node.cost + course_length

    def get_random_node(self):

        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = self.Node(random.uniform(self.min_rand_x, self.max_rand_x),
                            random.uniform(self.min_rand_y, self.max_rand_y),
                            random.uniform(-math.pi/4, math.pi/4)
                            )
        # TODO: Change range from -math.pi

        else:  # goal point sampling
            #self.end.yaw=random.uniform(-math.pi,math.pi)
            rnd = self.Node(self.end.x, self.end.y, self.end.yaw)

        return rnd

    def search_best_goal_node(self):

        goal_indexes = []
        for (i, node) in enumerate(self.node_list):
            if self.calc_dist_to_goal(node.x, node.y) <= self.goal_xy_th:
                goal_indexes.append(i)

        # angle check
        final_goal_indexes = []
        for i in goal_indexes:
            if abs(self.node_list[i].yaw - self.end.yaw) <= self.goal_yaw_th:
                pass
            final_goal_indexes.append(i)

        if not final_goal_indexes:
            return None

        min_cost = min([self.node_list[i].cost for i in final_goal_indexes])

        print("HEREyaw", [self.node_list[i].path_yaw for i in final_goal_indexes])

        for i in final_goal_indexes:
            if self.node_list[i].cost == min_cost:
                print("BEST",self.node_list[i].path_yaw)
                return i

        return None

    def generate_final_course(self, goal_index):
        print("final")
        path = [[self.end.x, self.end.y,self.end.yaw]]
        node = self.node_list[goal_index]
        while node.parent:
            for (ix, iy,iyaw) in zip(reversed(node.path_x), reversed(node.path_y),reversed(node.path_yaw)):
                path.append([ix, iy,iyaw])
                print(iyaw)
            node = node.parent
        path.append([self.start.x, self.start.y,self.start.yaw])
        print(path)
        return path


def main():
    print("Start rrt star with dubins planning")

    # ====Search Path with RRT====
    obstacleList = [
        (28.1, -16.5,20,10),
        (-31.9,-36.5,20,30),
        (-31.9,23.5,20,30)
    ]  # [x,y,size(radius)]

    # Set Initial parameters
    start = [0.0, 0.0, np.deg2rad(90.0)]
    goal = [34.25,33.10, np.deg2rad(90.0)]

    rrtstar_dubins = RRTStarDubins(start, goal, rand_area_x=[-5, 50.0],rand_area_y=[-45,45], obstacle_list=obstacleList)
    path = rrtstar_dubins.planning(animation=show_animation)

    # Draw final path
    x_val = []
    y_val = []
    for i in range(len(path)):
        x_val.append(path[i][0])
        y_val.append(path[i][1])

    if show_animation:  # pragma: no cover
        rrtstar_dubins.draw_graph(x_val=x_val,y_val=y_val)
        plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
        plt.scatter(x_val, y_val)
        plt.grid(True)

        plt.pause(0.001)


    plt.savefig('final.png')
    plt.show()


if __name__ == '__main__':
    main()