import torch, time, random

from math import pi as PI
from typing import List, Dict, Union, Callable

from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch_geometric.data import Data, Batch

from agents.experience import Experience


def polar(data: Data, norm=True, max_val=None) -> Data:
    (row, col), pos = data.edge_index, data.pos

    cart = pos[col] - pos[row]

    rho = torch.norm(cart, p=2, dim=-1).view(-1, 1)

    theta = torch.atan2(cart[..., 1], cart[..., 0]).view(-1, 1)
    theta = theta + (theta < 0).type_as(theta) * (2 * PI)

    if norm:
        rho = rho / (rho.max() if max_val is None else max_val)
        theta = theta / (2 * PI)

    polar = torch.cat([rho, theta], dim=-1)
    data.edge_attr = polar
    return data


class ReplayBuffer(Dataset):

    def __init__(self, buf_len: int):
        super(ReplayBuffer, self).__init__()
        self.memory = []
        self.__buf_len = buf_len
        self.__pointer = 0

    def __getitem__(self, idx: int) -> Experience:
        return self.memory[idx]

    def __len__(self) -> int:
        return len(self.memory)

    def push(self, e: Experience) -> None:
        if self.__pointer < self.__buf_len:
            self.memory.append(e)
        else:
            idx = self.__pointer % self.__buf_len
            self.memory[idx] = e
        self.__pointer += 1

    def overwrite(self, D: List[Experience]) -> None:
        self.clear()
        self.memory = D
        self.__pointer = len(D)

    def extend(self, D: List[Experience]) -> None:
        for e in D:
            self.push(e)

    def clear(self) -> None:
        self.memory = []
        self.__pointer = 0

    def sample(self, n: int) -> List[Experience]:
        assert n > 0
        return random.sample(self.memory, n)


class DoubleReplayBuffer:

    def __init__(self, b_1: ReplayBuffer, b_2: ReplayBuffer, batch_size: int, #ep_len: int, 
            loader_collate: Callable, is_frozen_1: bool=False, is_frozen_2: bool=False):
        self.b_1 = b_1 #RBDataLoader(b_1, ep_len, loader_collate, batch_size // 2)
        self.b_2 = b_2 #RBDataLoader(b_2, ep_len, loader_collate, batch_size // 2)
        self.b1_frozen = is_frozen_1
        self.b2_frozen = is_frozen_2
        self.batch_size = batch_size
        self.collate = loader_collate

    def sample(self) -> Dict[str, Union[Batch, torch.Tensor]]:
        X_1 = self.b_1.sample(self.batch_size//2) #next(iter(self.b_1))
        X_2 = self.b_2.sample(self.batch_size//2) #next(iter(self.b_2))
        '''
        src = Batch.from_data_list(X_1['src'].to_data_list() + X_2['src'].to_data_list())
        dest = Batch.from_data_list(X_1['dest'].to_data_list() + X_2['dest'].to_data_list())
        act = torch.cat([X_1['act'], X_2['act']], dim=-1)
        rew = torch.cat([X_1['r'], X_2['r']], dim=-1)
        return {
            'src': src, 'dest': dest, 'act': act, 'r': rew
        }
        '''
        return self.collate(X_1 + X_2)
        

    def push(self, e: Experience) -> None:
        if not self.b1_frozen:
            self.b_1.push(e)
        if not self.b2_frozen:
            self.b_2.push(e)
    

class RBDataLoader(DataLoader):

    def __init__(self, dataset: ReplayBuffer, ep_len: int, collate_fn: Callable, batch_size: int=1):
        super(RBDataLoader, self).__init__(
            dataset, 
            batch_size=batch_size, 
            collate_fn=collate_fn,
            sampler=RandomSampler(
                dataset, 
                replacement=True, 
                num_samples=ep_len * batch_size
            )
        )

    def sample(self) -> Dict[str, Union[Batch, torch.Tensor]]:
        return next(iter(self)) 