import time, math, os
import torch
import trimesh

from tqdm import tqdm

from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam

#from torch_geometric.utils import to_trimesh
from torch_geometric.data import Batch

from agents.prim.prim_state import PrimState
from agents.prim.prim_action import PrimAction
from agents.prim.prim_model import PrimModel
from agents.prim.prim_reward import PrimReward
from agents.prim.prim_expert import PrimExpert
from agents.environment import Environment
from agents.experience import Experience
from agents.replay_buffer import ReplayBuffer, RBDataLoader, polar
from agents.double_dqn_trainer import DoubleDQNTrainer
from agents.base_model import BaseModel
from agents.state import State
from agents.action import Action

from geometry.cuboid import Cuboid
from geometry.shape_dataset import ShapeDataset


'''
class TestDataset(Dataset):

    def __init__(self, num_rand: int, size: int):
        self.data = [torch.rand(size, size) for _ in range(num_rand)]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]
'''

def train():
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    PrimState.init_state_space(episode_len=300)
    x = torch.tensor([-1.0,-1.0,-1.0])
    y = torch.tensor([1.0,1.0,1.0])
    unit = torch.dist(x, y).item() / 16
    PrimAction.init_action_space(PrimState.num_primitives, 2, [-2 * unit, -unit, unit, 2 * unit])
    
    R = PrimReward(0.1, 0.01, device)
    env = Environment(PrimAction.ground(), R)

    online = PrimModel(PrimAction.act_space_size)
    target = PrimModel(PrimAction.act_space_size)
    target.load_state_dict(online.state_dict())
    target.eval()
    
    opt = Adam(online.parameters(), 0.0001)
    exp = PrimExpert(env)

    trainer = DoubleDQNTrainer(online, target, env, opt, exp, 0.9, 0.8, 20, 0.02, device)

    rfc = ShapeDataset('../data/ShapeNet', items_per_category={'watercraft': 600, 'plane': 800, 'pistol': 600, 'rocket': 800})
    imit = ShapeDataset('../data/ShapeNet', items_per_category={'watercraft': 600, 'plane': 800, 'pistol': 600, 'rocket': 800}, partition='IMIT')
    print("IMITATION DATA: " + str(len(imit)) + " instances")
    print("REINFORCEMENT DATA: " + str(len(rfc)) + " instances")
    t1 = time.time()
    trainer.train(rfc, imit, PrimState.episode_len, 200000, 100000, 64, 4, 1000, '../model/PRIM.pth')
    print("Training time: " + str(time.time() - t1))


def test():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    PrimState.init_state_space(episode_len=300)
    x = torch.tensor([-1.0,-1.0,-1.0])
    y = torch.tensor([1.0,1.0,1.0])
    unit = torch.dist(x, y).item() / 16
    PrimAction.init_action_space(PrimState.num_primitives, 2, [-2 * unit, -unit, unit, 2 * unit])

    R = PrimReward(0.1, 0.01, device)
    env = Environment(PrimAction.ground(), R)

    online = PrimModel(PrimAction.act_space_size)
    online.eval()
    online.to(device)

    online.load_state_dict(torch.load('../model/PRIM.pth'))

    rfc = ShapeDataset('../data/ShapeNet', items_per_category={'plane': 800, 'pistol': 600, 'rocket': 800}, partition='TEST')

    def select_action(m: PrimModel, s: State) -> Action:
        b = Batch.from_data_list([polar(s.to_geom_data())])
        pred = torch.argmax(m(b.to(device)), -1).item()
        action = env.get_action(pred)
        action.set_index(pred)
        return action

    curr = online.get_initial_state(rfc[-1]['reference'])
    for idx in range(PrimState.episode_len):
        if idx%30 == 0:
            curr.meshify().show()
        act = select_action(online, curr)
        print(act)
        curr = act(curr)


    
def virtual_expert_modeling():
    PrimState.init_state_space() 
    x = torch.tensor([-1.0,-1.0,-1.0])
    y = torch.tensor([1.0,1.0,1.0])
    unit = torch.dist(x, y).item() / 16
    PrimAction.init_action_space(PrimState.num_primitives, 2, [-2 * unit, -unit, unit, 2 * unit])

    R = PrimReward(0.1, 0.01)
    env = Environment(PrimAction.ground(), R)
    exp = PrimExpert(R, env)

    M = PrimModel(10, PrimAction.act_space_size)
    D = ShapeDataset('../data/ShapeNet', categories=['pistol'])
    import random
    idx = random.randint(1, 10)
    episode = D[idx]
    BaseModel.new_episode(episode['reference'], episode['mesh'])

    current = PrimState.initial()
    current.meshify()
    
    experiences = exp.unroll(current, 27 * 4)
    actions = [e.get_action() for e in experiences]

    for act in actions:
        current = act(current)

    print(len(current))
    t = torch.zeros(64 ** 3, dtype=torch.long, device=os.environ['DEVICE'])
    t[episode['mesh']] = 1
    print(R.iou(current, t))
    print(R.iou_sum(current, t))
        
    scene = trimesh.Scene()
    scene.add_geometry(current.meshify())
    scene.add_geometry(episode['target'])
    scene.show()



if __name__ == "__main__":

    # TODO: Figure why Expert relabeling needs more time than unrolling
    # TODO: Let the reinforcement() procedure sample from both the Long Term Memory and the Rfc buffer

    t = time.time()
    #virtual_expert_modeling()
    train()
    #test()
    print("Total time: " + str(time.time() - t))
