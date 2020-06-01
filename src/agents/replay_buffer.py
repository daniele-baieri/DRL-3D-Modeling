import torch

from math import pi as PI
from typing import List, Dict

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


def collate(x: List[Experience]) -> Dict:
    curr = [polar(e.get_source().to_geom_data()) for e in x]
    succ = [polar(e.get_destination().to_geom_data()) for e in x]
    return {
        'src': Batch.from_data_list(curr),
        'dest': Batch.from_data_list(succ),
        'act': torch.LongTensor([e.get_action().get_index() for e in x]),
        'r': torch.FloatTensor([e.get_reward() for e in x])
    }


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


class RBDataLoader(DataLoader):

    def __init__(self, dataset: ReplayBuffer, ep_len: int, batch_size: int=1):
        super(RBDataLoader, self).__init__(
            dataset, 
            batch_size=batch_size, 
            collate_fn=collate,
            sampler=RandomSampler(
                dataset, 
                replacement=True, 
                num_samples=ep_len * batch_size
            )
        )