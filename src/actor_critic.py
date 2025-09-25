import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F
from utils import select_device, parse_state, get_action_id, infer_flat


#environment
from environment import env



#Parameters
EPISODES = 10
MAX_STEPS_PER_EPISODE = 5
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
          nn.Linear(3136, 512),
          nn.ReLU(),
          nn.Linear(512, action_dim)
        )        


    def forward(self, state: float):
        return self.conv(state)


class Critic(nn.Module):
    def __init__(self, state_dim, device=None):
        super(Critic, self).__init__()
        c, h, w = state_dim
        self.features = nn.Sequential(
            nn.Conv2d(c, 32, 8, 4), 
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), 
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), 
            nn.ReLU(),
        )
        n_flat = infer_flat(self.features, state_dim, device=device)
        self.value = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_flat, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, state):
        return self.value(self.features(state)).squeeze(-1)

        


def train():
    print("Training...")
    device = select_device()
    state_dim = env.observation_space.shape
    actor = Actor(state_dim=state_dim, action_dim=env.action_space.n).float()
    critic = Critic(state_dim=state_dim, device=device).float()
    for episode in range(EPISODES):
        state = parse_state(env.reset(), device=device)
        episode_reward = 0
        actor.train()
        critic.train()
        for step in range(MAX_STEPS_PER_EPISODE):
            action_values = actor(state)
            action = get_action_id(action_values)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = truncated or terminated
            episode_reward += reward
            value = critic(state)



        
    
    
