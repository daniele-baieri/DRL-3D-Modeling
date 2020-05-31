import torch

from torch.utils.data import DataLoader
from torch.optim import Optimizer

from torch_geometric.data import Batch

from agents.base_model import BaseModel
from agents.environment import Environment
from agents.replay_buffer import ReplayBuffer, RBDataLoader
from agents.experience import Experience
from agents.action import Action
from agents.state import State
from agents.prim.prim_state import PrimState


class DoubleDQNTrainer:

    def __init__(self, online: BaseModel, target: BaseModel,
                 env: Environment, opt: Optimizer, 
                 buf: RBDataLoader):
        self.__online = online
        self.__target = target
        self.__target.eval()
        self.__env = env
        self.__opt = opt
        self.__rl_buf = buf

    def train(self, data: DataLoader, episode_len: int):
        
        self.__online.train()
        self.__online.set_episode_len(episode_len)
        self.__target.set_episode_len(episode_len)

        for episode in data:
            self.__env.set_state(PrimState.initial())
            self.__online.set_reference(episode)
            self.__target.set_reference(episode)

            for _ in range(episode_len):
                self.__online.step()
                self.__target.step()

                action = self.select_action(self.__env.get_state())
                succ, reward = self.__env.transition(action)
                exp = Experience(self.__env.get_state(), succ, action, reward)
                self.__rl_buf.push(exp)
                self.__env.set_state(succ)

                self.optimize_model()
            self.__target.load_state_dict(self.__online.state_dict())

    def warmup_il(self) -> None:
        raise NotImplementedError

    def optimize_model(self) -> None:
        exps = next(iter(self.__rl_buf))
        state_in = exps['src']

        # NOTE: can't proceed: model not defined
        # pred = ???


    def select_action(self, s: State) -> Action:
        b = Batch.from_data_list([s.to_geom_data()])
        return self.__env.get_action(
            torch.argmax(self.__online(b), -1).item()
        )