import time, math, os, random
import torch
import trimesh

from tqdm import tqdm

from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam, Adadelta, RMSprop

#from torch_geometric.utils import to_trimesh
from torch_geometric.data import Batch

from agents.prim.prim_state import PrimState
from agents.prim.prim_action import PrimAction
from agents.prim.prim_model import PrimModel
from agents.prim.prim_reward import PrimReward
from agents.prim.prim_expert import PrimExpert
from agents.imitation.long_short_memory import LongShortMemory
from agents.environment import Environment
from agents.experience import Experience
from agents.replay_buffer import ReplayBuffer, RBDataLoader, polar, DoubleReplayBuffer
from agents.double_dqn_trainer import DoubleDQNTrainer
from agents.base_model import BaseModel
from agents.state import State
from agents.action import Action

from geometry.cuboid import Cuboid
from geometry.shape_dataset import ShapeDataset
from geometry.voxelize import voxelize


def train():

    #torch.autograd.set_detect_anomaly(True)

    BATCH_SIZE = 64
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    PrimState.init_state_space(episode_len=300)
    PrimAction.init_action_space(PrimState.num_primitives, 2, [-2, -1, 1, 2])
    
    R = PrimReward(0.1, 0.01, device)
    env = Environment(PrimAction.ground(), R)

    online = PrimModel(PrimAction.act_space_size).to(device)
    params = sum(p.numel() for p in online.parameters() if p.requires_grad)
    print("MODEL PARAMETERS: " + str(params))
    target = PrimModel(PrimAction.act_space_size).to(device)
    target.load_state_dict(online.state_dict())
    target.eval()
    
    opt = Adam(online.parameters(), lr=0.0001)
    exp = PrimExpert(env)

    trainer = DoubleDQNTrainer(online, target, env, opt, exp, 0.9, 0.8, 4000, 0.02, device)

    rfc = ShapeDataset('../data/ShapeNet', items_per_category={'watercraft': 400, 'plane': 493, 'pistol': 307, 'car': 400})
    imit = ShapeDataset('../data/ShapeNet', items_per_category={'watercraft': 400, 'plane': 493, 'pistol': 307, 'car': 400}, partition='IMIT')
    print("IMITATION DATA: " + str(len(imit)) + " instances")
    print("REINFORCEMENT DATA: " + str(len(rfc)) + " instances")

    imit_buf = LongShortMemory(200000, PrimState.episode_len, BATCH_SIZE, PrimState.collate_prim_experiences)
    rfc_mem = ReplayBuffer(100000)
    rfc_buf = DoubleReplayBuffer(rfc_mem, imit_buf.long_memory, BATCH_SIZE, PrimState.collate_prim_experiences, is_frozen_2=True)
    t1 = time.time()
    trainer.train(rfc, imit, PrimState.episode_len, imit_buf, rfc_buf, 2, 2000, '../model/PRIM.pth')
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
   
    test = ShapeDataset('../data/ShapeNet', items_per_category={'car': 400}, partition='TEST')
    print("TEST DATA: " + str(len(test)) + " instances")
    data = test[random.randint(0, len(test))]
    data['target'].show()

    def select_action(m: PrimModel, s: State) -> Action:
        b = Batch.from_data_list([s.to_geom_data()])
        pred = torch.argmax(m(b.to(device)), -1).item()
        action = env.get_action(pred)
        action.set_index(pred)
        return action

    curr = online.get_initial_state(data['reference'])
    rew = 0.0
    for idx in range(PrimState.episode_len):
        #if idx%27 == 0:
        #    curr.meshify().show()
        act = select_action(online, curr)
        print(act)
        succ = act(curr)
        rew += R(curr, succ, data['mesh'])
        curr = succ
    scene = trimesh.scene.Scene()
    scene.add_geometry(data['target'])
    scene.add_geometry(curr.meshify())
    scene.show()

    __eval_iou(R, curr, data['mesh'], device)
    print("Accum. Rew.: "  + str(rew))    

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

    #online.load_state_dict(torch.load('../model/PRIM.pth'))

    test = ShapeDataset('../data/ShapeNet', items_per_category={'watercraft': 400, 'plane': 493, 'pistol': 307, 'car': 400}, partition='TEST')
    print("TEST DATA: " + str(len(test)) + " instances")
    data = test[random.randint(0, len(test))]
    #data['target'].show()
    env.set_target(data['mesh'])

    curr = online.get_initial_state(data['reference'])
    history = exp.unroll(curr, PrimState.episode_len)
    for idx in range(len(history)):
        curr = history[idx].get_destination()
        print(history[idx].get_action())
        #if idx%27 == 0:
        #    scene = trimesh.scene.Scene()
        #    scene.add_geometry(data['target'])
        #    scene.add_geometry(curr.meshify())
        #    scene.show()
    scene = trimesh.scene.Scene()
    scene.add_geometry(data['target'])
    scene.add_geometry(curr.meshify())
    scene.show()

    plot_voxels(curr)

    __eval_iou(R, curr, data['mesh'], device)

def unit_testing():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    PrimState.init_state_space(episode_len=300)
    PrimAction.init_action_space(PrimState.num_primitives, 2, [-2, -1, 1, 2])
    R = PrimReward(0.1, 0.01, device)
    env = Environment(PrimAction.ground(), R)


    import numpy as np
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    test = ShapeDataset('../data/ShapeNet', items_per_category={'plane': 493}, partition='TEST')
    state = PrimState.initial(test[0]['reference'])
    for i in range(100):
        state = env.get_random_action()(state)

    state.meshify().show()
    plot_voxels(state)

def plot_voxels(state):
    import numpy as np
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    #X, Y, Z = np.mgrid[-1:1:64j, -1:1:64j, -1:1:64j]
    T = state.voxelize(use_cuda=True).cpu().view(64,64,64).nonzero().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scat = ax.scatter(T[:, 0], T[:, 2], T[:, 1])
    fig.colorbar(scat, aspect=5)
    plt.show()

def __eval_iou(R, state, voxels, device):

    cubes_vox_new = state.voxelize(cubes=True, use_cuda=(device=='cuda')).bool()
    vox_new = cubes_vox_new.sum(dim=0).bool()

    print(R.iou(vox_new, voxels.to(device)))
    print(R.iou_sum(cubes_vox_new, voxels.to(device), len(state)))


if __name__ == "__main__":

    t = time.time()
    #virtual_expert_modeling()
    train()
    #test()
    #unit_testing()
    print("Total time: " + str(time.time() - t))
