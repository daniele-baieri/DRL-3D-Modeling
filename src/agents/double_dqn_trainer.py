import torch
from typing import Tuple, Dict, Union

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
from agents.expert import Expert
from agents.imitation.long_short_memory import LongShortMemory

from geometry.shape_dataset import ShapeDataset


class DoubleDQNTrainer:

    def __init__(self, online: BaseModel, target: BaseModel,
                 env: Environment, opt: Optimizer, exp: Expert,
                 buf: RBDataLoader, disc_fact: float):
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

        
    def imitation(self) -> None:
        
        # start = self.__exp.get_action_sequence(init_state, batch_size)
        # Starting from env.get_state(), generate M new Experiences D using the virtual expert
        # SHORT = D, LONG = D
        # for k = 1 to N:
        #     optimize_model(), but sampling is done equally on SHORT and LONG
        #     Using the current model predictions, get a series of new states S
        #     Annotate S with actions, rewards and successors given by the virtual expert (obtaining D')
        #     LONG = LONG U D', SHORT = D'
        N = self.__online.get_episode_len()

        for idx in range(N):
            batch = self.imitation_buffer.sample()

    def reinforcement(self) -> None:
        
        for _ in range(self.__online.get_episode_len()):

            action = self.select_action(self.__env.get_state())
            succ, reward = self.__env.transition(action)
            exp = Experience(self.__env.get_state(), succ, action, reward)
            self.reinforcement_buffer.dataset.push(exp)
            self.__env.set_state(succ)

            batch = next(iter(self.reinforcement_buffer))
            self.optimize_model(batch)

            self.__online.step()
            self.__target.step()
            

    def train(self, data: ShapeDataset, initial_state: State,
            long_mem: int, short_mem: int, rl_mem: int, batch_size: int) -> None:

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

            self.imitation()
            self.reinforcement()

            self.__target.load_state_dict(self.__online.state_dict())

    def optimize_model(self, batch: Dict[str, Union[Batch, torch.Tensor]]) -> None:
        # exps = next(iter(self.__rl_buf))
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