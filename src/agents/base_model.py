import torch
import trimesh
from trimesh import Trimesh
from torch.nn import Module


class BaseModel(Module):

    def __init__(self):
        super(BaseModel, self).__init__()
        #self._ref = None
        self._step = -1
        self._episode_len = -1

    @classmethod
    def new_episode(cls, new_ref: torch.FloatTensor, new_model: Trimesh) -> None:
        cls._ref = new_ref
        cls._model = new_model

    @classmethod
    def get_reference(cls) -> torch.FloatTensor: 
        return cls._ref#.clone().detach()

    @classmethod
    def get_model(self) -> Trimesh:
        return self._model

    def step(self) -> None:
        self._step += 1

    def get_step(self) -> int:
        return self._step

    def zero_step(self) -> int:
        self._step = 0

    def get_episode_len(self) -> int:
        return self._episode_len

    def set_episode_len(self, ep_len: int) -> int:
        assert ep_len > 0
        self._episode_len = ep_len

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError