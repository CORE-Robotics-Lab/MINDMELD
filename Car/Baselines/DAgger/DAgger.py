
import time
from MINDMELD.Car import Record as record
import os
from MINDMELD.Car.Base_Agent import Learner_Agent
import tkinter as tk
from MINDMELD.Car.config import *


class Dagger_Agent(Learner_Agent):

    def __init__(self, car, env_num, rollout_num, subj_num=None, agent_name="Dagger"):
        super().__init__(car, env_num, rollout_num, subj_num=subj_num, name=agent_name)
        self.dagger_steps = DAGGER_STEPS

    def record_corrective(self, tk_obj):
        """
            Records corrective demonstration
            Parameters:
                        tk_obj: TK GUI object

        """
        path_save_feedback = os.path.join(self.base_path, 'Data', self.agent_name, 'Feedback', 'subj_' +
                                              str(self.subj_num), 'env_' + str(self.env_num),
                                              'roll_out_' + str(self.rollout_num))
        if not os.path.exists(path_save_feedback):
            print("making directory")
            os.makedirs(os.path.join(path_save_feedback))
        else:
            print("directory already exists")

        print("READY")

        text_ready = tk.Label(tk_obj, text="Ready...", font=("Arial", 35), fg='red')
        text_ready.place(x=90, y=350)
        tk_obj.update()

        time.sleep(2)

        text_go = tk.Label(tk_obj, text="Go!", font=("Arial", 35), fg='green')
        text_go.place(x=90, y=450)
        tk_obj.update()

        record.record_actions(path_save_feedback, self.car)
        dist, _ = self.rollout_policy()

        self.car.client.stopRecording()
        record.stop_record()
        print("FINISHED")
        time.sleep(.5)
        record.moveRecording(path_save_feedback)
        time.sleep(.5)
        record.combine_corrective(path_save_feedback)


        text_finished = tk.Label(tk_obj, font=("Arial", 25), text="Finished corrective demonstration " +
                                                                  str(self.rollout_num+self.total_rollouts*self.env_num+1))
        text_finished.place(x=60, y=650)
        tk_obj.update()

        return text_ready, text_go, text_finished, dist

    def train(self, initial_only=False,test=False):
        """
            trains agent. If agent is mindmeld, train on Intitial and mapped feedback. If agent is Dagger, train on initial and original feedback.

        """
        print("AGENT NAME: ", self.agent_name)
        if initial_only:
            super().train(["Initial"])
        else:
            if test:
                super().train(["Initial", "Corrected_Feedback_Test"],save_name='Models_Test')
            elif self.agent_name == 'MeLD':
                super().train(["Initial", "Corrected_Feedback"])
            else:
                super().train(["Initial", "Feedback"])
