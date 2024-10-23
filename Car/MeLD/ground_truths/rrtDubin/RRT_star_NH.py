import pygame, random, math, numpy
from numpy import *
from pygame.locals import *
import matplotlib.path as mplPath
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import copy
import time


class RRTStarDubins():
    class Node:
        x = 0
        y = 0
        theta = 0
        parent = None
        list = []
        cost = 0
        steer=0


        def __init__(self, xcord, ycord, theta,steer):
            self.x = xcord
            self.y = ycord
            self.theta = theta
            self.steer=steer


    def __init__(self, draw=True, obstacle_list=[], search_space=[], goal=[], start=[],path="",plot_path=[],U_s=120,scale=5,xshift=40,yshift=40,prev_steering=0,phsi_orig=0):
        self.path=path
        self.scale = scale
        self.xshift = xshift * self.scale
        self.yshift = yshift * self.scale
        self.Xmax = search_space[2] * self.scale+self.xshift  # Window size
        self.Ymax = search_space[3] * self.scale+self.yshift
        self.Xmin = search_space[0] * self.scale+self.xshift
        self.Ymin = search_space[1] * self.scale+self.yshift
        print(self.Xmax, self.Xmin,self.Ymax,self.Ymin,self.xshift,self.yshift,search_space)
        self.L = 5*self.scale  # length of car
        self.U_s = U_s*self.scale # linear velocity
        self.phsi_orig=phsi_orig
        self.h = .05 # runge-kutta constant
        self.t=.7
        self.inc_ang=self.h*math.pi/4*self.t
        print("T",self.t)
        print("HHH",self.h)
        self.numnodes = 10000
        self.matplot_obstacles = copy.deepcopy(obstacle_list)
        self.obstacle_list = obstacle_list.copy()  # [x,y,size(radius)]
        self.draw = draw
        self.plot_path=plot_path
        print("speed",U_s)
        self.prev_steering=prev_steering

        # [x,y,size(radius)]

        for i in range(len(self.obstacle_list)):
            for j in range(len(self.obstacle_list[i])):
                temp = self.obstacle_list[i][j] * self.scale
                if j == 0:
                    temp += self.xshift
                elif j==1:
                    temp+=self.yshift
                self.obstacle_list[i][j] = temp

        self.obstacle_draw_list = self.obstacle_list

        self.goal_x = goal[0] * self.scale + self.xshift
        self.goal_y = goal[1] * self.scale + self.yshift
        self.start_x = start[0] * self.scale + self.xshift
        self.start_y = start[1] * self.scale + self.yshift
        self.adj=False
        #self.theta_start=start[2]-math.pi/4
        """if -math.pi/4<start[2]<math.pi/4:
            self.theta_start =start[2]-math.pi/4
            self.adj=math.pi/4
        elif start[2]<-math.pi/4:
            self.theta_start=start[2]-math.pi/4
            self.adj=0

        else:
            self.theta_start=start[2]
            self.adj=0
            """
        self.adj=0
        self.theta_start=start[2]-self.adj

        print("start",self.theta_start)

        self.d = 2

        if draw:
            self.Green = 124, 252, 0
            self.black = 0, 0, 0
            self.Red = 255, 0, 0
            self.Blue = 0, 0, 128
            self.white = 255, 255, 255
            pygame.init()  # pygame initialization
            self.screen = pygame.display.set_mode([1000, 1000])
            pygame.display.set_caption('RRT')
            self.screen.fill(self.white)
            for i in range(len(self.obstacle_draw_list)):
                pygame.draw.rect(self.screen, self.black, pygame.Rect(self.obstacle_draw_list[i]))

            pygame.draw.circle(self.screen, self.Red, [self.goal_x, self.goal_y], 5)
            #pygame.display.update()

    def dist(self, x1, y1, x2, y2):  # distance between two point
        D = math.sqrt(math.pow(x2 - x1, 2) + math.pow(y2 - y1, 2))
        return D

    def Range_kutta(self, qnear, phsi_goal):  # Runge Kutta integration method
        thet = qnear.theta
        qnew = self.Node(0, 0, 0,0)
        x = qnear.x
        y = qnear.y
        q_list = []
        h = self.h
        U_s = self.U_s
        L = self.L
        alltheta=0
        phsi=qnear.steer


        for i in range(1,int(self.t/self.h), 1):
            k1 = numpy.array([h * U_s * cos(thet), h * U_s * sin(thet), h * U_s * (numpy.tan(phsi) / L)])

            thet1 = thet + (k1[2] )
            x1 = x + (k1[0] )
            y1 = y + (k1[1] )
            pygame.draw.aaline(self.screen, self.black, [x1, y1], [x, y], 1)
            #pygame.display.update()

            x = x1
            y = y1
            alltheta+=(thet1-thet)
            #print(alltheta,self.U_s,self.h)
            thet = thet1


            if self.check_collision(x, y):
                qnew.x=Inf
                qnew.y=Inf
                qnew.theta=qnear.theta
                qnew.steer=phsi
                q_list.append(qnew)
                return q_list
            elif (x1 >= self.Xmin) & (y1 >= self.Ymin) & (x <= self.Xmax) & (y <= self.Ymax):
                qnew.x = x
                qnew.y = y
                qnew.theta = thet
                qnew.steer = phsi
                q_list.append(qnew)
            elif (x1 < self.Xmin) | (y1 < self.Ymin) | (x > self.Xmax) | (y > self.Ymax):
                qnew.x = Inf
                qnew.y = Inf
                qnew.theta = qnear.theta
                qnew.steer = phsi
                q_list.append(qnew)

            #incrementally update phsi because the wheels take time to actually turn
            if phsi_goal>0:
                if phsi+self.inc_ang>phsi_goal:
                    phsi=phsi_goal
                else:
                    phsi += self.inc_ang
            else:
                if phsi-self.inc_ang<phsi_goal:
                    phsi=phsi_goal
                else:
                    phsi -= self.inc_ang


        #print("len",len(q_list))
        return [qnew]

    def check_collision(self, x, y):
        for (v_x, v_y, w, h) in self.obstacle_list:

            list_x = (x < w + v_x)
            list_x2 = (x > v_x)

            list_y = (y < h + v_y)
            list_y2 = (y > v_y)

            if list_x and list_y and list_x2 and list_y2:
                return True

        return False



    def inte_right(self, qnear):  # integrate over line where turn angle is 45 degree(right)
        phsi = qnear.steer
        # assuming car can turn pi/4 in 1 second
        #low=phsi-math.pi/4
        #high=phsi+math.pi/4
        # assuming car can turn pi/4 in 1 second and t is timestep
        denominator = numpy.round(1/self.t*4)
        low=phsi-math.pi/denominator
        high=phsi+math.pi/denominator
        if low<-math.pi/4:
            low=-math.pi/4
        if high>math.pi/4:
            high=math.pi/4


        all_n=[]
        r=numpy.arange(low,high,self.h)

        for i in range(len(r)):

            n1 = self.Range_kutta(qnear, r[i])
            all_n+=n1


        return all_n

    def min_dist(self, q3, qrand):
        #best_q1 = q1[0]
        #best_q2 = q2[0]
        best_q3 = q3[0]
        best_dist1 = inf
        """for q in q1:

            d = self.dist(q.x, q.y, qrand.x, qrand.y)
            if d < best_dist1:
                best_dist1 = d
                best_q1 = q
        best_dist2 = inf
        for q in q2:

            d = self.dist(q.x, q.y, qrand.x, qrand.y)
            if d < best_dist2:
                best_dist2 = d
                best_q2 = q
                """
        best_dist3 = inf

        for q in q3:

            d = self.dist(q.x, q.y, qrand.x, qrand.y)
            if d < best_dist3:
                best_dist3 = d
                best_q3 = q
        """d_list = [best_dist1, best_dist2, best_dist3]
        D = min(d_list)
        if D == best_dist1:
            return best_q1, 1
        elif D == best_dist2:
            return best_q2, 2
        elif D == best_dist3:
            return best_q3, 3
            """
        return best_q3


    def runge_kutta_draw(self, qnear, phsi):  # integration method to draw a line over selected path through pygame
        thet = qnear.theta
        x = qnear.x
        y = qnear.y
        pygame.draw.circle(self.screen, self.Green, (qnear.x, qnear.y), 1)
        h = self.h
        U_s = self.U_s
        L = self.L

        for i in range(1, int(self.t/self.h), 1):
            k1 = numpy.array([h * U_s * cos(thet), h * U_s * sin(thet), h * U_s * (numpy.tan(phsi) / L)])


            #thet1 = thet + (k1[2] / 6 + k2[2] / 3 + k3[2] / 3 + k4[2] / 6) * h*self.t
            #x1 = x + (k1[0] / 6 + k2[0] / 3 + k3[0] / 3 + k4[0] / 6) * h*self.t
            #y1 = y + (k1[1] / 6 + k2[1] / 3 + k3[1] / 3 + k4[1] / 6) * h*self.t
            thet1 = thet + (k1[2] )
            x1 = x + (k1[0] )
            y1 = y + (k1[1] )

            if self.check_collision(x, y):
                return
            pygame.draw.aaline(self.screen, self.black, [x1, y1], [x, y], 1)
            #pygame.display.update()
            x = x1
            y = y1
            thet = thet1



    def attach_final_node(self, node, qgoal):
        D = []
        for n in node:
            tmpdist = self.dist(n.x, n.y, qgoal.x, qgoal.y)
            D.append(tmpdist)
        q_pre_final = node[D.index(min(D))]

        #q1, ang1 = self.inte_line(q_pre_final)
        #q2, ang2 = self.inte_left(q_pre_final)
        q3= self.inte_right(q_pre_final)
        [qnew, d] = self.min_dist( q3, qgoal)
        if self.draw:
            if d == 1:
                self.runge_kutta_draw(q_pre_final, 0)
            elif d == 2:
                self.runge_kutta_draw(q_pre_final, math.pi / 4)
            elif d == 3:
                self.runge_kutta_draw(q_pre_final, -math.pi / 4)
        qgoal.parent = node.index(q_pre_final)
        node.append(qgoal)
        return node

    def draw_path(self, node):
        for n in node:
            pygame.draw.aalines(self.screen, self.Blue, False, n.list)
        #pygame.display.update()

    def get_rand(self):
        r=random.randint(0, 100)
        if  r > 60:
            qxrand = random.uniform(self.Xmin, self.Xmax)
            qyrand = random.uniform(self.Ymin, self.Ymax)
        elif r > 40 and r < 60:
            # sample along straight line
            qxrand = random.uniform(self.goal_x, self.start_x)
            slope = (self.goal_y-self.start_y)/(self.goal_x - self.start_x)
            qyrand = slope*(qxrand-self.start_x)+self.start_y
        elif r < 40 and r > 20:
            qxrand = self.start_x
            qyrand = self.start_y
        else:
            qxrand = self.goal_x
            qyrand = self.goal_y

        qrand = self.Node(qxrand, qyrand, 0,0)
        return qrand

    def get_nearest(self, qrand, node):
        ndist = []
        for n in node:
            tmp = self.dist(n.x, n.y, qrand.x, qrand.y)
            ndist.append([tmp])
        qnear = node[ndist.index(min(ndist))]

        return qnear

    def steer(self, qnear, qrand):
        #q1, ang1 = self.inte_line(qnear)
        #q2, ang2 = self.inte_left(qnear)
        q3= self.inte_right(qnear)
        qnew = self.min_dist(q3, qrand)

        if self.draw:
            draw_node = self.Node(0, 0, 0,0)
            draw_node.x = qnear.x
            draw_node.y = qnear.y
            draw_node.theta = qnear.theta
            #if d == 1:
            #self.runge_kutta_draw(draw_node, draw_node.steer)
            """elif d == 2:
                self.runge_kutta_draw(draw_node, ang2)
            elif d == 3:
                self.runge_kutta_draw(draw_node, ang3)
                """
        return qnew

    def plot_arrow(self,x, y, yaw, length=10.0, width=0.5, fc="r",
                   ec="k"):  # pragma: no cover
        """
        Plot arrow
        """

        if not isinstance(x, float):
            for (i_x, i_y, i_yaw) in zip(x, y, yaw):
                self.plot_arrow(i_x, i_y, i_yaw)
        else:
            plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
                      fc=fc, ec=fc, head_width=width, head_length=width)
            plt.plot(x, y)

    def draw_graph(self, rnd=None, x_val=None, y_val=None,theta=None):
        plt.clf()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect('key_release_event',
                                     lambda event: [exit(0) if event.key == 'escape' else None])
        """if rnd is not None:
            plt.plot(rnd.x, rnd.y, "^k")
        for node in self.node_list:
            if node.parent:
                plt.plot(node.path_x, node.path_y, "-g")
                """

        for (x, y, w, h) in self.matplot_obstacles:
            # self.plot_circle(ox, oy, size)
            plt.gca().add_patch(Rectangle((y, x), h, w, linewidth=1, edgecolor='r', facecolor='none'))
        plt.plot((self.start_y - self.yshift) / self.scale, (self.start_x - self.xshift) / self.scale, "xr")
        plt.plot((self.goal_y - self.yshift) / self.scale, (self.goal_x - self.xshift) / self.scale, "xr")
        if not x_val is None:
            plt.plot(y_val, x_val, 'r')
        plt.plot(self.plot_path[1],self.plot_path[0])
        goal_ang=numpy.arctan((x_val[-2]-x_val[-1])/(y_val[-2]-y_val[-1]))
        if y_val[-2] < y_val[-1]:
            goal_ang+=math.pi
        self.plot_arrow(y_val[-1],x_val[-1],goal_ang)


        self.plot_arrow((self.start_y - self.yshift) / self.scale,  (self.start_x - self.xshift) / self.scale,math.pi/2-((self.theta_start)+self.adj),fc='g')

        #self.plot_arrow((self.start_y - self.yshift) / self.scale, (self.start_x - self.xshift) / self.scale,
        #            math.pi / 2 - ((self.theta_start) ), fc='g')
        plt.axis([(self.Ymin-self.yshift)/self.scale, (self.Ymax-self.yshift)/self.scale, (self.Xmin-self.xshift)/self.scale, (self.Xmax-self.xshift)/self.scale])
        plt.grid(True)
        # self.plot_start_goal_arrow()
        plt.pause(0.01)
        plt.savefig(self.path)
        #plt.show()
        # time.sleep(.2)

    def find_near_nodes(self, node, qnew):
        Qnearest = []
        r = 40.0

        for n in node:
            if (math.pow((n.x - qnew.x), 2) + (math.pow((n.y - qnew.y), 2)) - math.pow(r, 2)) <= 0:
                phsi_poss=numpy.arctan((qnew.theta-n.theta)*self.L/(self.t*self.U_s))
                if phsi_poss>-math.pi/4 and phsi_poss<math.pi/4:
                    Qnearest.append([n,phsi_poss])
        return Qnearest

    def rewire(self, node, qnew, qmin):

        for n in node:
            if n == qmin:
                qnew.parent = node.index(n)
        return node,qnew

    def get_path(self):
        node = []
        ##initialize goal and first node
        qinit = self.Node(self.start_x, self.start_y, self.theta_start,self.phsi_orig)
        qgoal = self.Node(self.goal_x, self.goal_y, math.pi / 4,0)
        qinit.parent = None
        node.append(qinit)
        for v in range(1, self.numnodes):

            qrand = self.get_rand()
            qnear = self.get_nearest(qrand, node)
            qnew = self.steer(qnear, qrand)

            qnew.cost = self.dist(qnew.x, qnew.y, qnear.x, qnear.y) + qnear.cost
            Qnearest = self.find_near_nodes(node, qnew)
            qmin = qnear
            action_min=qnew.steer
            Cmin = qnew.cost
            for Q_A in Qnearest:
                Q=Q_A[0]
                if Q.cost + self.dist(Q.x, Q.y, qnew.x, qnew.y) < Cmin:

                    qmin = Q
                    Cmin = Q.cost + self.dist(Q.x, Q.y, qnew.x, qnew.y)
                    action_min=Q.steer

                    if self.draw:
                        pygame.draw.aaline(self.screen, self.Green, [qnear.x, qnear.y], [qnew.x, qnew.y])
                        pygame.draw.aaline(self.screen, self.black, [qmin.x, qmin.y], [qnew.x, qnew.y])
                         #pygame.display.update()

            qnew.parent = node.index(qnear)
            node,qnew = self.rewire(node, qnew, qmin)
            qnew.steer=action_min
            node.append(qnew)
            #node=self.attach_final_node(node,qgoal)

            if v % 1000 == 0:
                print("num nodes: " + str(v))

        D = []

        for n in node:
            #if not qgoal==n:
            tmpdist = self.dist(n.x, n.y, qgoal.x, qgoal.y)
            D.append(tmpdist)
        qfinal_pos=[]
        for i in range(len(D)):
            if D[i]<20:
                qfinal_pos.append(node[i])
        if qfinal_pos==[]:
            qfinal=node[D.index(min(D))]
        else:
            bestI=0
            best_cost=1000000000000000000
            for k in range(len(qfinal_pos)):
                total_cost=0
                end=qfinal_pos[k]
                j=0
                while end.parent is not None:
                    j+=1
                    total_cost+=end.cost
                    sta=int(end.parent)
                    end=node[sta]
                    if j>500:
                        print("BREAK",node.index(end))
                        break
                if total_cost<best_cost:
                    best_cost=total_cost
                    bestI=k
            qfinal=qfinal_pos[bestI]

        node.append(qgoal)
        qgoal.parent = node.index(qfinal)
        print("qgoal.parent",qgoal.parent)
        if self.draw:
            pygame.draw.line(self.screen, self.black, [qgoal.x, qgoal.y], [qfinal.x, qfinal.y])
            pygame.display.update()
            #time.sleep(2)
        end = qgoal
        sta = qfinal

        x_list = []
        y_list = []
        path=[]
        theta_list=[]
        first_int=sta
        while end.parent is not None:
            sta = int(end.parent)
            if sta==first_int:
                "no solution"
                break
            if self.draw:
                pygame.draw.aaline(self.screen, self.Red, [end.x, end.y], [node[sta].x, node[sta].y])
                pygame.display.update()
            end = node[sta]
            path.append([(end.x - self.xshift) / self.scale,(end.y - self.yshift) / self.scale,end.theta,end.steer])
            x_list.append((end.x - self.xshift) / self.scale)
            y_list.append((end.y - self.yshift) / self.scale)
            theta_list.append(end.theta)
            first_int=sta

        self.draw_graph(x_val=x_list, y_val=y_list,theta=theta_list)
        pygame.image.save(self.screen, self.path[:-4]+"_all.png")
        if self.draw:
            for e in pygame.event.get():
                if e.type == QUIT or (e.type == KEYUP and e.key == K_ESCAPE):
                    sys.exit("Leaving because you requested it.")

        return path


def main():
    ol = [
        [22.5, -18.5, 10, 35],
        [-38.05, -39.75, 16, 24],
        [-31.9, 23.5, 20, 30],
        [12.01, 47.24, 10, 21]
    ]
    planner = RRTStarDubins(draw=False, obstacle_list=ol, search_space=[30, 30, 75, 75], goal=[34.25, 33.10],
                            start=[0, 0])
    x_path, y_path = planner.get_path()


if __name__ == '__main__':
    main()
    running = True
    while running:
        for e in pygame.event.get():
            if e.type == QUIT or (e.type == KEYUP and e.key == K_ESCAPE):
                sys.exit("Leaving because you requested it.")
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False