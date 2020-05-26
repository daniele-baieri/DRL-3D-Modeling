import torch
from torch.nn import Module


class BaseModel(Module):

    def __init__(self):
        self._ref = None
        self._step = -1

    def set_reference(self, new_ref: torch.FloatTensor) -> None:
        self._ref = new_ref

    def step(self) -> None:
        self._step += 1

    def get_step(self) -> int:
        return self._step

    def get_reference(self) -> torch.FloatTensor:
        return self._ref.clone().detach()

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError