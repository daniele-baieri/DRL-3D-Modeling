import torch
from torch.nn import Module


class BaseModel(Module):

    def __init__(self):
        super(BaseModel, self).__init__()
        self._ref = None
        self._step = -1
        self._episode_len = -1

    def set_reference(self, new_ref: torch.FloatTensor) -> None:
        self._ref = new_ref

    def get_reference(self) -> torch.FloatTensor: 
        return self._ref.clone().detach()

    def step(self) -> None:
        self._step += 1

    def get_step(self) -> int:
        return self._step

    def get_episode_len(self) -> int:
        return self._episode_len

    def set_episode_len(self, ep_len: int) -> int:
        assert ep_len > 0
        self._episode_len = ep_len

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError