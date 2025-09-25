import torch
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