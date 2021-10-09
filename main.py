from stable_baselines3 import SAC
from env import BaseEnv, ImageWrapper, GrayImageWrapper
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy

exp_id = "sac_gray"
env = BaseEnv()
env = ImageWrapper(env, resolution=50)
env = GrayImageWrapper(env)
check_env(env)
model = SAC("CnnPolicy", env, verbose=1, device='cuda')

model.learn(total_timesteps=10000, log_interval=4)
model.save(exp_id)

del model   # remove to demonstrate saving and loading

model = SAC.load(exp_id)
env = BaseEnv()
env = ImageWrapper(env, resolution=50)
evaluate_policy(model, env, render=True)
