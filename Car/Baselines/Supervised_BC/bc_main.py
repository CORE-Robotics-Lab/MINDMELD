import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from MINDMELD.Car.CarEnv import CarEnv
from MINDMELD.Car.Baselines.Supervised_BC.Supervised_BC import BC_Agent
import tkinter
import pandas as pd


class BC_Main_TK():
    def __init__(self, num_env, num_demos, subj_id):
        print("STARTING BC")
        self.top = tkinter.Tk()
        self.top.title('LFD')
        self.top.geometry('600x700')  # Size 200, 200
        self.car = CarEnv(calibration=False)
        self.car.reset(0)
        self.num_env = num_env
        self.num_demos = num_demos

        self.subj_num = subj_id
        self.agent = BC_Agent(self.car, 0, 0, subj_num=self.subj_num)

        self.i = 0
        self.j = 1
        self.k = 0

        self.save_path = os.path.join(self.agent.base_path, 'Data', self.agent.agent_name, 'Distance', 'subj_' +
                                      str(self.subj_num))
        if os.path.isfile(self.save_path):
            raise Exception("Distance Path already exists")
        else:
            if not os.path.exists(self.save_path):
                os.makedirs(os.path.join(self.save_path))
        self.distance = []

        self.rollout_flag = True
        self.demo_flag = False
        self.train_flag = True
        self.start_flag = True
        self.orig_color = self.top.cget("background")
        self.agent.save_train_num += 1

        self.begin()

    def start(self):
        """
         starts recording for demonstrations in environment self.i and rollout self.j
        """
        if self.rollout_flag and not self.demo_flag:
            self.top.update()
            self.agent.set_env_rollout(env_num=self.i, rollout_num=self.j)
            text_ready, text_go, text_finished = self.agent.record_initial(self.top)
            text_ready.destroy()
            text_go.destroy()

            if self.i == self.num_env - 1 and self.j == self.num_demos:
                f = pd.DataFrame({'distance': self.distance})
                f.to_csv(os.path.join(self.save_path, 'distance.csv'), index=False)
                self.top.destroy()
            if self.i == self.num_env - 1 and self.j == self.num_demos-1:
                self.startButton.destroy()
                self.start_flag = False
            if self.j == self.num_demos:
                self.j = 0
                self.i += 1
            else:
                self.j += 1
            self.rollout_flag = False
            self.train_flag = False
            self.demo_flag = True
            self.trainButton.configure(bg='green')
            if self.start_flag:
                self.startButton.configure(bg=self.orig_color)
        else:
            print("MUST TEST AGENT FIRST")

    def train(self):
        """
         trains agent on  demonstrations
        """
        if self.demo_flag:
            self.trainButton.configure(bg=self.orig_color)
            self.agent.train()
            self.demo_flag = False
            self.rolloutButton.configure(bg='green')
            self.train_flag = True
            print("FINISHED TRAINING")
        else:
            print("MUST PROVIDE DEMONSTRATION BEFORE TRAINING")

    def rollout(self):
        """
            rollouts current policy
           """
        if self.train_flag:
            self.rolloutButton.configure(bg=self.orig_color)
            self.car.reset(env_num=self.i)
            dist, _ = self.agent.rollout_policy()
            self.distance.append(dist)
            print("Distance to goal: ", dist)
            f = pd.DataFrame({'distance': self.distance})
            f.to_csv(os.path.join(self.save_path, 'distance.csv'), index=False)
            self.rollout_flag = True
            if self.start_flag:
                self.startButton.configure(bg='green')
        else:
            print("MUST TRAIN AGENT FIRST")

    def stop(self):
        """
         stops recording for corrective demo
        """
        f = pd.DataFrame({'distance': self.distance})
        f.to_csv(os.path.join(self.save_path, 'distance.csv'), index=False)
        self.top.destroy()

    def redo(self):
        """
         allows participant to redo demonstration
        """
        if self.demo_flag and not self.train_flag:
            self.j -= 1
            self.agent.redo_initial(self.i, self.j)
            self.rollout_flag = True
            self.demo_flag = False
            self.train_flag = False
            self.start()
        elif not self.demo_flag:
            print("CANNOT REDO WITH NO DEMONSTRATION")
        elif self.train_flag:
            print("CANNOT REDO DEMONSTRATION AFTER TRAINING")

    def begin(self):
        """
         sets up and starts GUI for initial demonstration
        """
        self.startButton = tkinter.Button(self.top, height=2, width=20, text="Start Demo", font=("Arial", 25),
                                          command=self.start)

        self.rolloutButton = tkinter.Button(self.top, height=2, width=20, text="Test Agent", font=("Arial", 25),
                                            command=self.rollout)

        self.stopButton = tkinter.Button(self.top, height=1, width=20, text="Move on to next agent", font=("Arial", 25),
                                         command=self.stop)

        self.trainButton = tkinter.Button(self.top, height=1, width=20, text="Train agent", font=("Arial", 25),
                                          command=self.train)
        self.redoButton = tkinter.Button(self.top, height=1, width=20, text="Redo demonstration", font=("Arial", 25),
                                         command=self.redo)

        self.startButton.pack()
        self.trainButton.pack()
        self.rolloutButton.pack()
        self.redoButton.pack()
        self.stopButton.pack()
        self.rolloutButton.configure(bg='green')
        print("before mainloop")
        self.top.mainloop()
        print("after main loop")


# m=BC_Main_TK(1,10)

"""car=CarEnv(calibration=False)


num_env=1
num_rollouts=10

subj_num = input("Enter subject number: ")
agent=BC_Agent(car,0,0,subj_num=subj_num)
for i in range(num_env):
    for j in range(num_rollouts):
        agent.set_env_rollout(env_num=i, rollout_num=j)
        agent.record_initial()
        agent.train()
        car.reset(env_num=i)
        agent.set_env_rollout(env_num=i, rollout_num=j)
        enter = input("Press Enter to start rollout: ")
        agent.rollout_policy()
#agent.train()
#agent.rollout_policy()
"""
