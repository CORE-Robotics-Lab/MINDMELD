import os
from MINDMELD.Car.CarEnv import CarEnv
from MINDMELD.Car.Baselines.Supervised_BC.Supervised_BC import BC_Agent
import tkinter
import pandas as pd


class Initial_Demo():
    def __init__(self, num_env, num_corrective, num_initial, subj_id):
        print("STARTING Initial Demo")
        self.top = tkinter.Tk()
        self.top.title('LFD')
        self.top.geometry('600x700')  # Size 200, 200
        self.car = CarEnv(calibration=False)
        self.car.reset(0)
        self.num_env = num_env
        self.num_corrective = num_corrective
        self.num_initial = num_initial

        self.subj_num = subj_id
        self.agent = BC_Agent(self.car, 0, 0, subj_num=self.subj_num)
        self.agent.all_agents = True

        self.i = 0
        self.j = 0
        self.k = 0
        self.demo_flag = False
        self.train_flag = False
        self.orig_color = self.top.cget("background")

        self.distance = []
        self.begin()

    def start(self):
        """
         starts recording for intitial demo in environment self.i and rollout self.j
        """
        if self.j < self.num_initial:
            if not self.demo_flag:
                self.startButton.configure(bg=self.orig_color)
                self.top.update()
                self.agent.set_env_rollout(env_num=self.i, rollout_num=self.j)
                print("recording initial")
                text_ready, text_go, text_finished = self.agent.record_initial(self.top)

                text_ready.destroy()
                text_go.destroy()

                self.j += 1
                text_finished.destroy()
                self.train_flag = False
                self.demo_flag = True
                self.trainButton.configure(bg='green')
        else:
            print("done with initial")

            self.top.destroy()

    def redo(self):
        """
         allows participant to redo initial demonstration
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

    def train(self):
        """
         trains agent based upon initial demonstration
        """
        if self.demo_flag:
            self.trainButton.configure(bg=self.orig_color)
            self.redoButton.destroy()
            self.trainButton.destroy()
            self.agent.train()
            self.demo_flag = False
            self.train_flag = True
            print("FINISHED TRAINING")

            self.top.destroy()

        else:
            print("MUST PROVIDE DEMONSTRATION BEFORE TRAINING")
        self.train_flag = True

    def begin(self):
        """
         sets up and starts GUI for initial demonstration
        """

        self.startButton = tkinter.Button(self.top, height=2, width=20, text="Start", font=("Arial", 25),
                                          command=self.start)

        self.trainButton = tkinter.Button(self.top, height=1, width=20, text="Train agent", font=("Arial", 25),
                                          command=self.train)
        self.redoButton = tkinter.Button(self.top, height=1, width=20, text="Redo demonstration", font=("Arial", 25),
                                         command=self.redo)
        self.startButton.pack()
        self.trainButton.pack()
        self.redoButton.pack()
        self.startButton.configure(bg="green")
        self.top.mainloop()
