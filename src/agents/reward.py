import torch
from agents.state import State


class Reward:

    def __init__(self):
        pass

    def _forward(self, old: State, new: State) -> float:
        raise NotImplementedError

    def __call__(self, old: State, new: State, model: torch.LongTensor) -> float:
        return self._forward(old, new, model)