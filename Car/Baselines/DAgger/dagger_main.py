import os
from MINDMELD.Car.CarEnv import CarEnv
from MINDMELD.Car.Baselines.DAgger.DAgger import Dagger_Agent
import tkinter
import pandas as pd


class Dagger_Main_TK():
    def __init__(self, num_env, num_corrective, num_initial, subj_id):
        print("STARTING DAGGER")
        self.top = tkinter.Tk()
        self.top.title('LFD')
        self.top.geometry('600x700')  # Size 200, 200
        self.car = CarEnv(calibration=False)
        self.car.reset(0)
        self.num_env = num_env
        self.num_corrective = num_corrective
        self.num_initial = num_initial

        self.subj_num = subj_id
        self.agent = Dagger_Agent(self.car, 0, 0, subj_num=self.subj_num)

        self.i = 0
        self.j = 0
        self.k = 0
        self.demo_flag = False
        self.train_flag = True
        self.start_flag = True
        self.orig_color = self.top.cget("background")
        self.save_path = os.path.join(self.agent.base_path, 'Data', self.agent.agent_name, 'Distance', 'subj_' +
                                      str(self.subj_num))
        if os.path.isfile(self.save_path):
            print("Distance Path already exists")
        else:
            if not os.path.exists(self.save_path):
                os.makedirs(os.path.join(self.save_path))

        self.distance = []
        self.agent.save_train_num += self.num_initial
        self.begin()

    def start(self):
        """
         starts recording for corrective demo in environment self.i and rollout self.j
        """

        if self.train_flag:
            self.startButton.configure(bg=self.orig_color)
            self.agent.set_env_rollout(env_num=self.i, rollout_num=self.k)
            text_ready, text_go, text_finished, dist = self.agent.record_corrective(self.top)
            self.distance.append(dist)
            f = pd.DataFrame({'distance': self.distance})
            f.to_csv(os.path.join(self.save_path, 'distance.csv'), index=False)
            print("Distance to goal: ", dist)
            text_ready.destroy()
            text_go.destroy()

            if self.i == self.num_env-1 and self.k == self.num_corrective:
                self.startButton.destroy()
                text_ready = tkinter.Label(self.top, text="Finshed Training Agent", font=("Arial", 35), fg='red')
                text_ready.place(x=90, y=450)
                self.top.update()
                self.rolloutButton = tkinter.Button(self.top, height=2, width=20, text="Test Agent", font=("Arial", 25),
                                                    command=self.test_agent)
                self.rolloutButton.pack()
                self.rolloutButton.config(bg="green")
            elif self.k == self.num_corrective-1:
                self.startButton.destroy()
                self.start_flag = False
                self.rolloutButton = tkinter.Button(self.top, height=2, width=20, text="Test Agent", font=("Arial", 25),
                                                    command=self.test_agent)
                self.rolloutButton.pack()
                self.j = 0

            if self.k == self.num_corrective - 1:
                self.k = 0
                self.i += 1
            else:
                self.k += 1
            self.trainButton.configure(bg="green")
            self.train_flag = False
            self.demo_flag = True

    def stop(self):
        """
         stops recording for corrective demo
        """
        print("STOPPING DAgger")
        f = pd.DataFrame({'distance': self.distance})
        f.to_csv(os.path.join(self.save_path, 'distance.csv'), index=False)

        self.top.destroy()

    def test_agent(self):
        if self.train_flag:
            print("in dagger rollout")
            self.car.reset(env_num=self.i-1)
            dist, _ = self.agent.rollout_policy()
            self.distance.append(dist)
            f = pd.DataFrame({'distance': self.distance})
            f.to_csv(os.path.join(self.save_path, 'distance.csv'), index=False)
            self.top.destroy()
            print("Distance to goal", dist)
        else:
            print("MUST TRAIN FIRST")

    def rollout(self):
        """
         rollouts current policy
        """
        self.car.reset(env_num=self.i-1)
        dist, _ = self.agent.rollout_policy()
        self.distance.append(dist)
        f = pd.DataFrame({'distance': self.distance})
        f.to_csv(os.path.join(self.save_path, 'distance.csv'), index=False)
        self.top.destroy()
        print("Distance to goal", dist)

    def redo(self):
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
         trains agent on intitial and corrective demonstrations
        """
        if self.demo_flag:
            self.trainButton.configure(bg=self.orig_color)
            self.agent.train()
            self.demo_flag = False
            self.train_flag = True
            print("FINISHED TRAINING")
            if self.start_flag:
                self.startButton.configure(bg="green")

        else:
            print("MUST PROVIDE DEMONSTRATION BEFORE TRAINING")
        self.train_flag = True

    def begin(self):
        """
         sets up and starts GUI for initial demonstration
        """

        self.startButton = tkinter.Button(self.top, height=2, width=20, text="Start", font=("Arial", 25),
                                          command=self.start)
        self.stopButton = tkinter.Button(self.top, height=1, width=20, text="Move on to next agent", font=("Arial", 25),
                                         command=self.stop)
        self.trainButton = tkinter.Button(self.top, height=1, width=20, text="Train agent", font=("Arial", 25),
                                          command=self.train)

        self.startButton.pack()
        self.trainButton.pack()
        self.stopButton.pack()
        self.startButton.configure(bg="green")
        print("before mainloop")
        self.top.mainloop()
        print("after main loop")

