from torch.utils.data import Dataset
from agents.experience import Experience


class ReplayBuffer(Dataset):

    def __init__(self, buf_len: int):
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