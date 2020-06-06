import torch
from typing import Tuple, Dict

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
        self.__rl_buf = buf
        self.__gamma = disc_fact
        self.__loss = SmoothL1Loss()

    def __sample(self, protocol: str) -> Dict:
        if protocol == 'R':
            return next(iter(self.__rl_buf))
        elif protocol == 'I':
            S = next(iter(self.__short_mem))
            L = next(iter(self.__long_mem))
            # TODO: fix this horrible thing.
            # Recall that you can't 'join' two pytorch_geometric Batches.

        
    def imitation(self, init_state: State, imit_steps: int, 
            batch_size: int, long_mem: int, short_mem: int) -> None:
        assert batch_size % 2 == 0

        self.__long_mem = RBDataLoader(
            ReplayBuffer(long_mem), imit_steps, batch_size=batch_size/2
        )
        self.__short_mem = RBDataLoader(
            ReplayBuffer(short_mem), imit_steps, batch_size=batch_size/2
        )

        start = self.__exp.get_action_sequence(init_state, batch_size)

        # TODO: LongShortMemory

        # Starting from env.get_state(), generate M new Experiences D using the virtual expert
        # SHORT = D, LONG = D
        # for k = 1 to N:
        #     optimize_model(), but sampling is done equally on SHORT and LONG
        #     Using the current model predictions, get a series of new states S
        #     Annotate S with actions, rewards and successors given by the virtual expert (obtaining D')
        #     LONG = LONG U D', SHORT = D'

        raise NotImplementedError

    def reinforcement(self, data: Dataset, initial_state: State):
        
        self.__online.train()

        for episode in data:
            self.__env.set_state(initial_state)
            BaseModel.new_episode(episode['mesh'], episode['reference'])
            #self.__online.new_episode()
            #self.__target.new_episode(episode['mesh'], episode['reference'])
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