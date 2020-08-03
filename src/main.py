import time, math, os, random
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



def train():
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    PrimState.init_state_space(episode_len=300)
    PrimAction.init_action_space(PrimState.num_primitives, 2, [-2, -1, 1, 2])
    
    R = PrimReward(0.1, 0.01, device)
    env = Environment(PrimAction.ground(), R)

    online = PrimModel(PrimAction.act_space_size)
    params = sum(p.numel() for p in online.parameters() if p.requires_grad)
    print("MODEL PARAMETERS: " + str(params))
    target = PrimModel(PrimAction.act_space_size)
    target.load_state_dict(online.state_dict())
    target.eval()
    
    opt = Adam(online.parameters(), 0.0001)
    exp = PrimExpert(env)

    trainer = DoubleDQNTrainer(online, target, env, opt, exp, 0.9, 0.8, 1000, 0.02, device)

    rfc = ShapeDataset('../data/ShapeNet', items_per_category={'watercraft': 600, 'plane': 800, 'pistol': 600, 'rocket': 800})
    imit = ShapeDataset('../data/ShapeNet', items_per_category={'watercraft': 600, 'plane': 800, 'pistol': 600, 'rocket': 800}, partition='IMIT')
    print("IMITATION DATA: " + str(len(imit)) + " instances")
    print("REINFORCEMENT DATA: " + str(len(rfc)) + " instances")
    t1 = time.time()
    trainer.train(rfc, imit, PrimState.episode_len, 200000, 100000, 64, 3, 1000, '../model/PRIM.pth')
    print("Training time: " + str(time.time() - t1))


def test():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    PrimState.init_state_space(episode_len=300)
    PrimAction.init_action_space(PrimState.num_primitives, 2, [-2, -1, 1, 2])

    R = PrimReward(0.1, 0.01, device)
    env = Environment(PrimAction.ground(), R)

    online = PrimModel(PrimAction.act_space_size)
    online.eval()
    online.to(device)

    online.load_state_dict(torch.load('../model/PRIM.pth'))

    test = ShapeDataset('../data/ShapeNet', items_per_category={'watercraft': 600, 'plane': 800, 'pistol': 600, 'rocket': 800}, partition='TEST')
    print("TEST DATA: " + str(len(test)) + " instances")
    data = test[random.randint(0, len(test))]
    data['target'].show()

    def select_action(m: PrimModel, s: State) -> Action:
        b = Batch.from_data_list([polar(s.to_geom_data())])
        pred = torch.argmax(m(b.to(device)), -1).item()
        action = env.get_action(pred)
        action.set_index(pred)
        return action

    curr = online.get_initial_state(data['reference'])
    for idx in range(PrimState.episode_len):
        if idx%27 == 0:
            curr.meshify().show()
        act = select_action(online, curr)
        print(act)
        curr = act(curr)
    curr.meshify().show()

    state_voxelized = curr.voxelize(cubes=True, use_cuda=device=='cuda')
    print(R.iou(state_voxelized.sum(dim=0), data['mesh'].to(device)))
    print(R.iou_sum(state_voxelized, data['mesh'].to(device), len(curr)))

   
def virtual_expert_modeling():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    PrimState.init_state_space(episode_len=300)
    #x = torch.tensor([-1.0,-1.0,-1.0])
    #y = torch.tensor([1.0,1.0,1.0])
    #unit = torch.dist(x, y).item() / 16
    PrimAction.init_action_space(PrimState.num_primitives, 2, [-2, -1, 1, 2])

    R = PrimReward(0.1, 0.01, device)
    env = Environment(PrimAction.ground(), R)
    exp = PrimExpert(env)

    online = PrimModel(PrimAction.act_space_size)
    online.eval()
    online.to(device)

    online.load_state_dict(torch.load('../model/PRIM.pth'))

    test = ShapeDataset('../data/ShapeNet', items_per_category={'watercraft': 600, 'plane': 800, 'pistol': 600, 'rocket': 800}, partition='TEST')
    print("TEST DATA: " + str(len(test)) + " instances")
    data = test[random.randint(0, len(test))]
    data['target'].show()
    env.set_target(data['mesh'])

    curr = online.get_initial_state(data['reference'])
    history = exp.unroll(curr, PrimState.episode_len)
    for idx in range(len(history)):
        curr = history[idx].get_source()
        print(history[idx].get_action())
        if idx%27 == 0:
            curr.meshify().show()
    curr.meshify().show()

    state_voxelized = curr.voxelize(cubes=True, use_cuda=device=='cuda')
    print(R.iou(state_voxelized.sum(dim=0), data['mesh'].to(device)))
    print(R.iou_sum(state_voxelized, data['mesh'].to(device), len(curr)))


if __name__ == "__main__":

    #TODO: pickle Imitation and Reinforcement buffer to allow for training interrupt/restart

    t = time.time()
    #virtual_expert_modeling()
    train()
    #test()
    print("Total time: " + str(time.time() - t))
