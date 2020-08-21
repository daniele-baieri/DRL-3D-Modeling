import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from typing import Tuple, Dict, Union, List
from tqdm import tqdm

from torch.utils.data import Dataset
from torch.nn import SmoothL1Loss, MSELoss
from torch.optim import Optimizer

from torch_geometric.data import Batch

from agents.base_model import BaseModel
from agents.environment import Environment
from agents.replay_buffer import ReplayBuffer, RBDataLoader, polar, DoubleReplayBuffer
from agents.experience import Experience
from agents.action import Action
from agents.state import State
from agents.imitation.expert import Expert
from agents.imitation.long_short_memory import LongShortMemory

from geometry.shape_dataset import ShapeDataset


IMITATION_ITER = "DAgger({}) // Loss: {:0.4f} + {:0.4f} = {:0.4f} ({:0.4f})"
REINFORCEMENT_ITER = "DDQN // Loss: {:0.4f} -- Reward: {:0.4f} -- Norm: {:0.4f}"
UNROLL_ITER = "Unrolling Agent // Acc. Reward: {}"


class DoubleDQNTrainer:

    def __init__(self, online: BaseModel, target: BaseModel, env: Environment, 
                 opt: Optimizer, exp: Expert, disc_fact: float, exp_margin: float,
                 update_frequency: int, eps_greedy: float, device: str):
        assert disc_fact >= 0.0 and disc_fact <= 1.0 
        assert exp_margin >= 0.0 and exp_margin <= 1.0 
        assert eps_greedy >= 0 and eps_greedy <= 1.0
            
        self.__opt_counter = 0
        self.__env = env
        self.__opt = opt
        self.__exp = exp
        self.__gamma = disc_fact
        self.__tau = update_frequency
        self.__epsilon = eps_greedy
        self.__margin = exp_margin
        self.__device = device
        self.__online = online#.to(self.__device)
        self.__target = target#.to(self.__device)
        self.__target.eval()
        self.__loss = MSELoss().to(self.__device)
        self.__accum_loss = 0.0
        
    def imitation(self, iterations: int, updates: int, episode_len: int) -> None:
        """
        The DAgger procedure.
        """

        initial = self.__env.get_state()
        
        for idx in range(iterations):  
            self.__accum_loss = 0.0
            rfc_loss = 0.0
            imit_loss = 0.0
            if idx == 0:
                D = self.__exp.unroll(initial, episode_len)
            else:
                new_episode = self.unroll(initial, episode_len)
                D = self.__exp.relabel(new_episode)
            self.imitation_buffer.aggregate(D)

            progress_bar = tqdm(range(updates), desc=IMITATION_ITER.format(idx, 0.0, 0.0, 0.0, 0.0))
            for upd in progress_bar:
                batch = self.imitation_buffer.sample()
                r, i, l, n = self.optimize_model(batch, joint=True)
                self.__accum_loss += l
                rfc_loss += r
                imit_loss += i
                progress_bar.set_description(IMITATION_ITER.format(idx, rfc_loss / (upd + 1), imit_loss / (upd + 1), self.__accum_loss / (upd+1), n))
            
            if idx == iterations - 1:
                _ = self.unroll(initial, episode_len)
            
    def reinforcement(self, episode_len: int) -> None:
        """
        Double DQN training algorithm: a single episode.
        """

        self.__accum_loss = 0.0
        accum_rew = 0.0
        progress_bar = tqdm(range(episode_len), desc=REINFORCEMENT_ITER.format(0.0, 0.0, 0.0))

        for step in progress_bar:
        
            curr = self.__env.get_state()
            self.__online.eval()
            action = self.select_action(curr)
            succ, reward = self.__env.transition(action)
            accum_rew += reward
            exp = Experience(curr, succ, action, reward)
            self.reinforcement_buffer.push(exp)

            batch = self.reinforcement_buffer.sample()
            l, n = self.optimize_model(batch)
            self.__accum_loss += l
            progress_bar.set_description(REINFORCEMENT_ITER.format(self.__accum_loss / (step+1), accum_rew, n))

    def train(self, rfc_data: ShapeDataset, imit_data: ShapeDataset, episode_len: int,
            #long_mem: int, rl_mem: int, batch_size: int, 
            imit_buffer: LongShortMemory, rfc_buffer: DoubleReplayBuffer,
            dagger_iter: int, dagger_updates: int, dump_path: str) -> None:
        """
        Main training loop. Trains a Q-network using the Double DQN algorithm, 
        with a first phase of imitation learning using the DAgger algorithm.
        """
        #assert short_mem == episode_len
        self.__model_path = dump_path

        #self.imitation_buffer = LongShortMemory(long_mem, episode_len, batch_size)
        self.imitation_buffer = imit_buffer
        self.reinforcement_buffer = rfc_buffer
        
        ep = 1
        for episode in imit_data:
            print("Imitation episode: " + str(ep))
            initial_state = self.__online.get_initial_state(episode['reference'])
            self.__env.set_state(initial_state)
            self.__env.set_target(episode['mesh'])

            self.imitation(dagger_iter, dagger_updates, episode_len)
            ep += 1

        self.__target.load_state_dict(self.__online.state_dict())
        #torch.save(self.__online.state_dict(), self.__model_path)
        torch.save(self.__online.state_dict(), self.__model_path + '.imitation_only')

        #rl_mem = ReplayBuffer(rl_mem)
        #self.reinforcement_buffer = DoubleReplayBuffer(rl_mem, self.imitation_buffer.long_memory, episode_len, batch_size, is_frozen_2=True)


        ep = 1
        for episode in rfc_data:
            print("Reinforcement episode: " + str(ep))
            initial_state = self.__online.get_initial_state(episode['reference'])
            self.__env.set_state(initial_state)
            self.__env.set_target(episode['mesh'])

            self.reinforcement(episode_len)
            ep += 1
        torch.save(self.__online.state_dict(), self.__model_path)

    def optimize_model(self, batch: Dict[str, Union[Batch, torch.Tensor]], joint: bool=False) -> Tuple[float, float, float]:

        if self.__opt_counter % self.__tau == 0:
            self.__target.load_state_dict(self.__online.state_dict())
            torch.save(self.__online.state_dict(), self.__model_path)
        self.__opt_counter += 1

        state_in = batch['src'].to(self.__device)
        next_in = batch['dest'].to(self.__device)
        action_ids = batch['act'].to(self.__device).unsqueeze(0).T
        rewards = batch['r'].to(self.__device).unsqueeze(-1)

        self.__online.train()
        model_out = self.__online(state_in) # Q-values for all actions
        pred = model_out.gather(1, action_ids) # Q-values for actions taken in experiences in the batch

        self.__online.eval()
        next_best_actions = self.__online(next_in).max(dim=-1)[1] # Best Q-values for next state actions   
        next_val = self.__target(next_in).gather(1, next_best_actions.unsqueeze(0).T) # Target net Q-values for best next state actions
        exp_act_val = (self.__gamma * next_val) + rewards

        loss = ((exp_act_val.detach() - pred) ** 2).mean()
        rfc_loss = loss.item()

        if joint:
            margins = torch.zeros_like(model_out, device=self.__device) + self.__margin # (margin) ^ {A x A}
            margins[range(len(action_ids)), action_ids.T] = 0 # margin is zero for expert Actions (i.e. action_ids in an imitation learning context)
            best_q = (model_out + margins).max(dim=-1)[0] # Find best Q-values, eventually including margin if the action is not expert
            diff = (best_q - pred.squeeze().detach()).mean() # Difference with Q-values of expert actions: if only expert actions are taken, this is 0
            loss += diff

        self.__opt.zero_grad()
        loss.backward()
        #for param in self.__online.parameters():
        #    param.grad.data.clamp(-1, 1)
        norm = nn.utils.clip_grad_norm_(self.__online.parameters(), 5.0)
        self.__opt.step()

        if joint:
            return rfc_loss, diff.item(), loss.item(), norm

        return rfc_loss, norm

    def select_action(self, s: State) -> Action:
        r = random.random()
        if r >= self.__epsilon: # greedy policy         
            b = Batch.from_data_list([s.to_geom_data()])
            pred = torch.argmax(self.__online(b.to(self.__device)), -1).item()
            action = self.__env.get_action(pred)
            action.set_index(pred)
        else: # random policy
            action = self.__env.get_random_action()
        return action

    def unroll(self, s: State, episode_len: int) -> List[Experience]:
        self.__online.eval()
        res = []
        self.__env.set_state(s)
        accum_reward = 0.0
        iterations = tqdm(range(episode_len), desc=UNROLL_ITER.format(accum_reward))
        for _ in iterations:
            curr = self.__env.get_state()
            act = self.select_action(curr)
            succ, r = self.__env.transition(act)
            accum_reward += r
            exp = Experience(curr, succ, act, r)
            res.append(exp)
            iterations.set_description(UNROLL_ITER.format(accum_reward))
        return res