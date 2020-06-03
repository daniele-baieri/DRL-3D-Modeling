import torch
from typing import Tuple

from torch.utils.data import Dataset
from torch.nn import SmoothL1Loss
from torch.optim import Optimizer

from torch_geometric.data import Batch

from agents.base_model import BaseModel
from agents.environment import Environment
from agents.replay_buffer import ReplayBuffer, RBDataLoader, polar
from agents.experience import Experience
from agents.action import Action
from agents.state import State
from agents.prim.prim_state import PrimState


class DoubleDQNTrainer:

    def __init__(self, online: BaseModel, target: BaseModel,
                 env: Environment, opt: Optimizer, 
                 buf: RBDataLoader, disc_fact: float):
        assert disc_fact >= 0.0 and disc_fact <= 1.0
        self.__online = online
        self.__target = target
        self.__target.eval()
        self.__env = env
        self.__opt = opt
        self.__rl_buf = buf
        self.__gamma = disc_fact
        self.__loss = SmoothL1Loss()
        
    def imitation_warmup(self) -> None:
        raise NotImplementedError

    def train(self, data: Dataset):
        
        self.__online.train()

        for episode in data:
            self.__env.set_state(PrimState.initial())
            self.__online.set_reference(episode)
            self.__target.set_reference(episode)
            self.__online.zero_step()
            self.__target.zero_step()

            for _ in range(self.__online.get_episode_len()):

                action = self.select_action(self.__env.get_state())
                succ, reward = self.__env.transition(action)
                exp = Experience(self.__env.get_state(), succ, action, reward)
                self.__rl_buf.dataset.push(exp)
                self.__env.set_state(succ)

                self.optimize_model()

                self.__online.step()
                self.__target.step()
            self.__target.load_state_dict(self.__online.state_dict())

    def optimize_model(self) -> None:
        exps = next(iter(self.__rl_buf))
        state_in = exps['src']
        next_in = exps['dest']
        action_ids = exps['act'].unsqueeze(0).t()
        rewards = exps['r']

        pred = self.__online(state_in).gather(1, action_ids)
        next_val = self.__target(next_in).max(dim=-1)[0].detach()
        exp_act_val = (self.__gamma * next_val) + rewards

        loss = self.__loss(pred, exp_act_val.unsqueeze(-1))
        print("Current loss: "+str(loss.item()))
        self.__opt.zero_grad()
        loss.backward()
        self.__opt.step()

    def select_action(self, s: State) -> Action:
        b = Batch.from_data_list([polar(s.to_geom_data())])
        pred = torch.argmax(self.__online(b), -1).item()
        action = self.__env.get_action(pred)
        action.set_index(pred)
        return action