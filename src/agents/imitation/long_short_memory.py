import torch
from typing import List, Dict, Union

from torch_geometric.data import Batch

from agents.experience import Experience
from agents.replay_buffer import ReplayBuffer, RBDataLoader


class LongShortMemory:

    def __init__(self, long_size: int, short_size: int, ep_len: int, batch_size: int):
        self.long_memory = ReplayBuffer(long_size)
        self.short_memory = ReplayBuffer(short_size)
        self.short_loader = RBDataLoader(self.short_memory, ep_len, batch_size // 2)
        long_batch = batch_size // 2 if batch_size % 2 == 1 else batch_size // 2 + 1
        self.long_loader = RBDataLoader(self.long_memory, ep_len, long_batch)

    def aggregate(self, D: List[Experience]) -> None:
        self.short_memory.overwrite(D)
        self.long_memory.extend(D)

    def sample(self) -> Dict[str, Union[Batch, torch.Tensor]]:
        S = next(iter(self.short_loader))
        L = next(iter(self.long_loader))
        src = Batch.from_data_list(S['src'].to_data_list() + L['src'].to_data_list())
        dest = Batch.from_data_list(S['dest'].to_data_list() + L['dest'].to_data_list())
        act = torch.cat([S['act'], L['act']], dim=-1)
        rew = torch.cat([S['r'], L['r']], dim=-1)
        return {
            'src': src, 'dest': dest, 'act': act, 'r': rew
        }