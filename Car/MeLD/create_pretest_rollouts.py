from MINDMELD.Car.CarEnv import CarEnv
from MINDMELD.Car.MeLD import pretest as pretest
from MINDMELD.Car.MeLD.pretest import Pretest


car=CarEnv()

num_envs = 4
num_rollouts = 4

for i in range( num_envs):
     for j in range( num_rollouts):
         pretest.save_pretest(car,i,j)
         pretest.get_gt(i, j)
         pretest_agent = Pretest(car, i, j)
         pretest_agent.rollout(i, j)


pretest.smooth_actions()









