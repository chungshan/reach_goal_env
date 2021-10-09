from stable_baselines3 import SAC
from env import BaseEnv, ImageWrapper, GrayImageWrapper, ObstacleWrapper
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
exp_id = "sac_color_obstacle"

env = BaseEnv()
env = ObstacleWrapper(env, False)
env = ImageWrapper(env, resolution=50)
check_env(env)
model = SAC("CnnPolicy", env, verbose=1, device='cuda')

model.learn(total_timesteps=10000, log_interval=4)
model.save(exp_id)

del model   # remove to demonstrate saving and loading

model = SAC.load(exp_id)
env = BaseEnv()
env = ImageWrapper(env, resolution=50, back_ground=(100, 0, 100))
re_mean, re_std = evaluate_policy(model, Monitor(env), render=True)
print(re_mean, re_std)