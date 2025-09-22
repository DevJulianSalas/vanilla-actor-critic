import torch
import torch.nn as nn
import torch.nn.functional as F


#environment
from environment import env



#Parameters
HIDDEN_DIM = 256
EPISODES = 1000




class Actor(nn.Module):
    def __init__(self, state_dim: float, action_dim: float, action_low: float, action_high: float):
        super(Actor, self).__init__()

        #actions
        self.action_dim = action_dim
        self.action_low = torch.tensor(action_low)
        self.action_high = torch.tensor(action_high)

        #scale and bias
        self.action_scale = (self.action_high - self.action_low) / 2.0
        self.action_bias = (self.action_low + self.action_high) / 2.0

        #hidden layers
        self.layer_1 = nn.Linear(state_dim, HIDDEN_DIM) 
        self.layer_2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)

        #output layers
        self.mean_layer = nn.Linear(HIDDEN_DIM, self.action_dim)
        self.log_std_layer = nn.Linear(HIDDEN_DIM, self.action_dim)



    def forward(self, state: float):
        x = F.relu(self.layer1(state))
        return x





def train():
    print("Training...")
    for episode in range(EPISODES):
        state = env.reset()
        print(state)
        
    
    
