#environment
import gym
import torch
from gym.spaces import Box
from gym.wrappers import FrameStack
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from torchvision import transforms as T

import gym_super_mario_bros
import numpy as np



#define environment
#SuperMarioBros-<world>-<stage>-v<version>
#world = 1 to 8
#stage = 1 to 4
#version = 0 to 4 -> indicating the ROM mode to use (0 is the original)
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0', apply_api_compatibility=True)
env = JoypadSpace(env, SIMPLE_MOVEMENT)


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip: int):
        super(SkipFrame, self).__init__(env)
        self._skip = skip
    
    def step(self, action):
        total_reward = 0.0
        for i in range(self._skip):
            print(action)
            obs, reward, done, trunk, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, trunk, info


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super(GrayScaleObservation, self).__init__(env)
        obj_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obj_shape, dtype=np.uint8)
    
    def permute_observation(self, observation):
        observation = np.transpose(observation, (2,0,1))
        return torch.tensor(observation.copy(), dtype=torch.float)
    
    def observation(self, observation):
        observation = self.permute_observation(observation)
        observation = T.Grayscale()(observation)
        observation = observation.squeeze(0)
        observation = observation / 255.0
        return observation

class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super(ResizeObservation, self).__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)
        self.observation_space = Box(low=0, high=255, shape=self.shape, dtype=np.uint8)

    
    def observation(self, observation):
        obs = observation.unsqueeze(0)
        obs = T.Resize(self.shape, antialias=True)(obs)
        obs = obs.squeeze(0)
        return obs
    

env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84)
env = FrameStack(env, num_stack=4)