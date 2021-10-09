import gym
import numpy as np
import matplotlib.pyplot as plt
import cv2


class BaseEnv(gym.Env):
    def __init__(self):
        self.map_size = np.array([10, 10])
        self.observation_space = gym.spaces.Box(low=-self.map_size[0], high=self.map_size[0], shape=(2,),
                                                dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.obs = np.array([1, 1])
        self.goal = np.array([9, 9])
        self.max_dist = np.sqrt(np.sum(self.map_size ** 2))
        self.step_ = 0
        self._max_step = 150
        self.fig, self.ax = None, None

    def step(self, action):
        done = False
        reached = False
        self.step_ += 1
        self.obs = self.obs + action
        result = np.logical_and(self.obs < self.map_size, self.obs > 0)

        if False in result:
            reward = -10
            done = True
        else:
            dist = np.sqrt(np.sum((self.goal - self.obs) ** 2))
            if dist < 0.01:
                reward = 10
                reached = True
                done = True
            else:
                reward = np.exp(1 - dist / self.max_dist)

        if self.step_ == self._max_step:
            done = True

        return self.obs, reward, done, {"episode": None, "reached": reached}

    def reset(self):
        self.step_ = 0
        self.obs = np.array([0, 0])

        return self.obs

    def render(self, mode="human"):
        if self.ax is None:
            self.fig, self.ax = plt.subplots()

        self.ax.clear()
        self.ax.scatter(self.obs[0], self.obs[1], marker='8')
        self.ax.scatter(self.goal[0], self.goal[1], marker='*')
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 10)
        plt.draw()
        plt.pause(0.001)


class ImageWrapper(gym.Wrapper):
    def __init__(self, env, resolution):
        gym.Wrapper.__init__(self, env)
        self.resolution = resolution
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(3, resolution, resolution), dtype=np.uint8)
        self.img_state = None

    def reset(self):
        obs = self.env.reset()
        return self.get_img(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        return self.get_img(obs), reward, done, info

    def get_img(self, obs):
        obs = self.obs_to_pixel(obs)
        goal = self.obs_to_pixel(self.env.goal)
        img_state = np.zeros((self.resolution, self.resolution, 3), np.uint8)
        img_state.fill(255)  # create white img
        img_state = cv2.circle(img_state, tuple(obs), 2, (255, 0, 0), -1)
        self.img_state = cv2.drawMarker(img_state, tuple(goal), (0, 0, 255), cv2.MARKER_STAR, 10)
        return self.img_state.copy().transpose(2, 0, 1)

    def obs_to_pixel(self, obs):
        pixel_x = int((obs[0] / self.env.map_size[0]) * self.resolution)
        pixel_y = self.resolution - int((obs[1] / self.env.map_size[0]) * self.resolution)

        return [pixel_x, pixel_y]

    def render(self, mode="human", **kwargs):

        cv2.imshow("Render", cv2.resize(self.img_state, (256, 256), interpolation=cv2.INTER_AREA))
        cv2.waitKey(1)


class GrayImageWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(1, self.env.resolution, self.env.resolution), dtype=np.uint8)
        self.gray_img_state = None

    def reset(self):
        obs = self.env.reset()
        self.gray_img_state = np.expand_dims(np.mean(obs, axis=0), axis=0).astype(np.uint8)

        return self.gray_img_state

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.gray_img_state = np.expand_dims(np.mean(obs, axis=0), axis=0).astype(np.uint8)
        return self.gray_img_state, reward, done, info

    def render(self, mode="human", **kwargs):
        cv2.imshow("Render", cv2.resize(self.gray_img_state.copy().transpose(1, 2, 0), (256, 256), interpolation=cv2.INTER_AREA))
        cv2.waitKey(1)