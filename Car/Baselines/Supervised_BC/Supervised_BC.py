import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Base_Agent import Learner_Agent


class BC_Agent(Learner_Agent):
    def __init__(self, car, env_num, rollout_num, subj_num=None):
        super().__init__(car, env_num, rollout_num, subj_num=subj_num, name="BC")

    def train(self):
        super().train(["Initial"])


