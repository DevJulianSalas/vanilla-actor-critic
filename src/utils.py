import torch
def select_device():
    return torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else
        "cpu"
    )


def parse_state(state):
     print(state[0])
     return state[0].__array__() if isinstance(state, tuple) else state.__array__()
