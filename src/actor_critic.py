import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F
from utils import select_device, parse_state


#environment
from environment import env



#Parameters
EPISODES = 5000
MAX_STEPS_PER_EPISODE = 1600
HIDDEN_DIM = 256
LOG_STD_MIN = -20
LOG_STD_MAX = 2



class Actor(nn.Module):
    def __init__(self, state_dim, action_dim=None):
        super(Actor, self).__init__()
        c, h, w = state_dim
        self.conv = nn.Sequential(
          nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
          nn.ReLU(),
          nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
          nn.ReLU(),
          nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
          nn.ReLU(),
          nn.Flatten(),
          nn.Linear(3200, 512),
          nn.ReLU(),
          nn.Linear(512, action_dim)
        )        


    def forward(self, state: float):
        print(state)
        return self.conv(state)

        
        


def train():
    print("Training...")
    state_dim = env.observation_space.shape
    actor = Actor(state_dim=state_dim, action_dim=env.action_space.n)
    device = select_device()
    for episode in range(EPISODES):
        state, info = env.reset()
        state = parse_state(state)
        state = torch.tensor(state, device=device).unsqueeze(0)
        episode_reward = 0
        actor.train()
        for step in range(MAX_STEPS_PER_EPISODE):
            action = actor(state)
            # next_state, reward, terminated, truncated, info = env.step(action)
            # done = truncated or terminated
            # episode_reward += reward

        
    
    
