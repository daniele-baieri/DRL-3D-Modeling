import torch
from torch.utils.data import DataLoader
from agents.base_model import BaseModel
from agents.environment import Environment
from agents.replay_buffer import ReplayBuffer
from agents.prim.prim_state import PrimState


class RLTrainer:

    def __init__(self, model: BaseModel, data: DataLoader, env: Environment):
        self.__data = data
        self.__model = model
        self.__env = env

    def train(self):
        for batch in self.__data:
            self.__env.set_state(PrimState.initial())

    def warmup_il(self) -> None:
        raise NotImplementedError

    def optimize_model(self) -> None:
        raise NotImplementedError