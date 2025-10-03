import torch
import gym
import matplotlib.pyplot as plt
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

# import gymnasium.wrappers



def select_device():
    return torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else
        "cpu"
    )


def parse_state(state, device):
     state = state[0].__array__() if isinstance(state, tuple) else state.__array__() 
     return torch.tensor(state, device=device).unsqueeze(0)


def get_action_id(action):
     return torch.argmax(action, axis=1).item()

def infer_flat(features, state, device, dtype=torch.float32):
    c, h, w = state
    with torch.no_grad():
          x = torch.zeros(1, c, h, w, device=device, dtype=dtype)
          y = features(x)
          n_flat = y.flatten(1).shape[1]
    return n_flat

def plot_rewards(all_episode_rewards, env_name, figsize=(10,5)):
    plt.figure(figsize=figsize)
    plt.plot(all_episode_rewards, label='Episode Reward', alpha=0.6)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title(f"Actor-critic vanilla training convergence ({env_name})")
    plt.legend
    plt.grid(True)
    plt.show()


def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def record_environment():
    env_record = gym_super_mario_bros.make(
          'SuperMarioBros-1-1-v0', 
          render_mode='rgb_array',
          apply_api_compatibility=True
    )
    env_record = JoypadSpace(env_record, SIMPLE_MOVEMENT)
    env_record = gym.wrappers.RecordVideo(
        env_record, 
        video_folder='videos/',
        episode_trigger=lambda x: x % 50 == 0,
        name_prefix='mario-agent'
    )
    return env_record

     