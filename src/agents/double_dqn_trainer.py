import torch
from typing import Tuple, Dict, Union, List

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
from agents.imitation.expert import Expert
from agents.imitation.long_short_memory import LongShortMemory

from geometry.shape_dataset import ShapeDataset


class DoubleDQNTrainer:

    def __init__(self, online: BaseModel, target: BaseModel,
                 env: Environment, opt: Optimizer, exp: Expert,
                 disc_fact: float):
        assert disc_fact >= 0.0 and disc_fact <= 1.0
        self.__online = online
        self.__target = target
        self.__target.eval()
        self.__env = env
        self.__opt = opt
        self.__exp = exp
        # self.__rl_buf = buf
        self.__gamma = disc_fact
        self.__loss = SmoothL1Loss()

        
    def imitation(self, iterations: int, updates: int) -> None:
        """
        The DAgger procedure.
        """

        initial = self.__env.get_state()

        N = self.__online.get_episode_len()
        start = self.__exp.unroll(initial, N)
        self.imitation_buffer.aggregate(start)

        for idx in range(iterations):
            for upd in range(updates):
                batch = self.imitation_buffer.sample()
                self.optimize_model(batch)
            new_episode = self.unroll(initial)
            D = self.__exp.relabel(new_episode)
            self.imitation_buffer.aggregate(D)
            
    def reinforcement(self) -> None:
        """
        Double DQN training algorithm: a single episode.
        """
        
        for _ in range(self.__online.get_episode_len()):

            curr = self.__env.get_state()
            action = self.select_action(curr)
            succ, reward = self.__env.transition(action)
            exp = Experience(curr, succ, action, reward)
            self.reinforcement_buffer.dataset.push(exp)

            batch = next(iter(self.reinforcement_buffer))
            self.optimize_model(batch)

            self.__online.step()
            self.__target.step()

    def train(self, data: ShapeDataset, initial_state: State,
            long_mem: int, short_mem: int, rl_mem: int, batch_size: int,
            dagger_iter: int, dagger_updates: int) -> None:
        """
        Main training loop. Trains a neural network using the Double DQN algorithm, 
        with a first phase of imitation learning using the DAgger algorithm.
        """

        ep_len = self.__online.get_episode_len()
        self.imitation_buffer = LongShortMemory(long_mem, short_mem, ep_len, batch_size)
        rl_mem = ReplayBuffer(rl_mem)
        self.reinforcement_buffer = RBDataLoader(rl_mem, ep_len, batch_size)
        
        self.__online.train()

        for episode in data:
            self.__env.set_state(initial_state)
            BaseModel.new_episode(episode['reference'], episode['mesh'])
            self.__online.zero_step()
            self.__target.zero_step()

            self.imitation(dagger_iter, dagger_updates)
            self.reinforcement()

            self.__target.load_state_dict(self.__online.state_dict())

    def optimize_model(self, batch: Dict[str, Union[Batch, torch.Tensor]]) -> None:
        state_in = batch['src']
        next_in = batch['dest']
        action_ids = batch['act'].unsqueeze(0).T
        rewards = batch['r']

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

    def unroll(self, s: State) -> List[Experience]:
        res = []
        self.__env.set_state(s)
        for _ in range(self.__online.get_episode_len()):
            curr = self.__env.get_state()
            act = self.select_action(curr)
            succ, r = self.__env.transition(act)
            exp = Experience(curr, succ, act, r)
            res.append(exp)
        return res