import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical
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
GAMMA = 0.99
ENTROPY_WEIGHT = 1e-3 
LR_ACTOR = 1e-5



class Actor(nn.Module):
    def __init__(self, state_dim, action_dim=None, device=None):
        super(Actor, self).__init__()
        c, h, w = state_dim
        self.conv = nn.Sequential(
          nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
          nn.ReLU(),
          nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
          nn.ReLU(),
          nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
          nn.ReLU(),
        )        
        n_flat = infer_flat(self.conv, state_dim, device)
        self.policy_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_flat, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )

    def forward(self, state: float):
        z = self.features(state)
        logits = self.policy_head(z)
        dist = Categorical(logits=logits)
        a = dist.sample() # a ~ π_θ(·|s_t)
        logp = dist.log_prob(a) # log π_θ(a_t|s_t)
        entropy = dist.entropy()  # H(π_θ(·|s_t))
        return logp, entropy, a


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
    actor = Actor(state_dim=state_dim, action_dim=env.action_space.n, device=device).float()  #π_θ(a|s) logits
    critic = Critic(state_dim=state_dim, device=device).float()
    #optimizers
    optimizer_actor = torch.optim.Adam(actor.parameters, lr=LR_ACTOR)
    for episode in range(EPISODES):
        state = parse_state(env.reset(), device=device)
        episode_reward = 0
        actor.train()
        critic.train()
        for step in range(MAX_STEPS_PER_EPISODE):
            #actor policy forward
            logp, entropy, action = actor(state) # π_θ(·|s_t) a_t ~ π_θ(·|s_t)
            
            next_state, reward, terminated, truncated, info = env.step(action)
            done = truncated or terminated
            episode_reward += reward
            
            # #tensor conversion
            reward_tensor = torch.tensor(episode_reward, device=device)
            next_state = parse_state(next_state, device=device)
            mask = torch.tensor(1 - done, device=device, dtype=torch.float32)
            
            value_critic = critic(state) #v = V_φ(s_t)
            with torch.no_grad():
                next_value = critic(next_state)  # V_φ(s_{t+1})
                td_target = reward_tensor + GAMMA * next_value * mask # td targetg y_t = r_t + γ V(s_{t+1})
            
            #error
            td_error = td_target - value_critic  #δ_t = y_t - V(s_t)
            #losses
            actor_loss = - (logp * td_error.detach()) - ENTROPY_WEIGHT * entropy


            #optimization
            optimizer_actor.zero_grad()
            actor_loss.backward()
            optimizer_actor.step()




        
    
    
