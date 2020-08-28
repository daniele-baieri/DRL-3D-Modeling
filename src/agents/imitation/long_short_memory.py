import torch
from typing import List, Dict, Union, Callable

from torch_geometric.data import Batch

from agents.experience import Experience
from agents.replay_buffer import ReplayBuffer, RBDataLoader


class LongShortMemory:

    def __init__(self, long_size: int, short_size: int, batch_size: int, loader_collate: Callable):
        """
        @param short_size: size of the short memory. Must be equal to the episode length
        """

        self.long_memory = ReplayBuffer(long_size)
        self.short_memory = ReplayBuffer(short_size)
        '''
        self.short_loader = RBDataLoader(self.short_memory, ep_len, loader_collate, batch_size // 2)
        long_batch = batch_size // 2 if batch_size % 2 == 0 else batch_size // 2 + 1
        self.long_loader = RBDataLoader(self.long_memory, ep_len, loader_collate, long_batch)
        '''
        self.batch_size = batch_size
        self.collate = loader_collate

    def aggregate(self, D: List[Experience]) -> None:
        self.short_memory.overwrite(D)
        self.long_memory.extend(D)

    def sample(self) -> Dict[str, Union[Batch, torch.Tensor]]:
        S = self.short_memory.sample(self.batch_size//2) #next(iter(self.short_loader))
        L = self.long_memory.sample(self.batch_size//2)# next(iter(self.long_loader))
        '''
        src = Batch.from_data_list(S['src'].to_data_list() + L['src'].to_data_list())
        dest = Batch.from_data_list(S['dest'].to_data_list() + L['dest'].to_data_list())
        act = torch.cat([S['act'], L['act']], dim=-1)
        rew = torch.cat([S['r'], L['r']], dim=-1)
        return {
            'src': src, 'dest': dest, 'act': act, 'r': rew
        }
        '''
        return self.collate(S + L)