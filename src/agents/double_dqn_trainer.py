import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from agents.base_model import BaseModel
from agents.environment import Environment
from agents.replay_buffer import ReplayBuffer
from agents.experience import Experience
from agents.action import Action
from agents.state import State
from agents.prim.prim_state import PrimState


class DoubleDQNTrainer:

    def __init__(self, online: BaseModel, target: BaseModel,
                 env: Environment, opt: Optimizer, 
                 buf: ReplayBuffer, opt_batch_size: int):
        self.__online = online
        self.__target = target
        self.__target.eval()
        self.__env = env
        self.__opt = opt
        self.__rl_buf = buf
        self.__replay_dl = DataLoader(self.__rl_buf, shuffle=True, batch_size=opt_batch_size)

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
        exps = next(iter(self.__replay_dl))
        state_in = torch.stack([e.get_source().tensorize() for e in exps])

        # NOTE: can't proceed: model not defined
        # pred = ???


    def select_action(self, s: State) -> Action:
        #NOTE: I didn't write this. Check it after defining the model
        return self.__env.get_action(
            self.__online(s.tensorize()).max(1)[1].view(1, 1) 
        )