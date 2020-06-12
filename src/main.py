import time, math
import torch
import trimesh

from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam

from torch_geometric.utils import to_trimesh

from trimesh.repair import fix_inversion, fix_normals, fix_winding

from agents.prim.prim_state import PrimState
from agents.prim.prim_action import PrimAction
from agents.prim.prim_model import PrimModel
from agents.prim.prim_reward import PrimReward
from agents.environment import Environment
from agents.experience import Experience
from agents.replay_buffer import ReplayBuffer, RBDataLoader
from agents.double_dqn_trainer import DoubleDQNTrainer
from agents.base_model import BaseModel

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


def test():

    PrimState.init_state_space(4, 2)
    
    prims = int(math.pow(PrimState.num_primitives, 3))
    PrimAction.init_action_space(prims, 2, [0.5, 1.0, 1.5])
    
    R = PrimReward(0.1, 0.001)
    env = Environment(PrimAction.ground(), R)

    online = PrimModel(10, PrimAction.act_space_size)
    target = PrimModel(online.get_episode_len(), PrimAction.act_space_size)
    target.load_state_dict(online.state_dict())
    target.eval()
    
    opt = Adam(online.parameters(), 0.01)
    rep = ReplayBuffer(10)
    rb = RBDataLoader(rep, online.get_episode_len(), 4)

    trainer = DoubleDQNTrainer(online, target, env, opt, rb, 0.999)

    ds = TestDataset(10, 120)

    t1 = time.time()
    trainer.train(ds)
    print("Training time: " + str(time.time() - t1))
'''
    
    
if __name__ == "__main__":

    print("Begin")
    PrimState.init_state_space(4, 64)
    PrimAction.init_action_space(PrimState.num_primitives, 2, [-0.1, -0.25, 0.1, 0.25])
    s = PrimState.initial()
    print("Initialized")
    a1 = PrimAction(0, vert=0, axis=0, slide=0.25)
    a2 = PrimAction(1, vert=0, axis=0, slide=0.1)
    a3 = PrimAction(2, vert=1, axis=1, slide=0.25)
    a4 = PrimAction(8, delete=True)
    a5 = PrimAction(0, vert=0, axis=1, slide=0.25)
    s1 = a1(a2(a3(a4(a5(s)))))
    print("Computed successor")

    D = ShapeDataset('/home/bayo/Documents/CS1920/DLAI-s2-2020/project/data/ShapeNet', categories=['rifle'])
    ref = torch.rand(2,2)
    M = next(iter(D))['mesh']

    #print(M.shape)
    #print(s1.voxelize(64).shape)

    BaseModel.new_episode(ref, M)


    #m = s.meshify()
    R = PrimReward(0.2, 0.1)
    t = time.time()
    print(R(s, s1))
    print("Reward time: " + str(time.time() - t))

    t = time.time()
    #test()
    print("Test time: " + str(time.time() - t))

    # old tests
    """
    PrimState.init_state_space(4, 2)
    PrimAction.init_action_space(64, 2, [0.5, 1])
    a1 = PrimAction(0, vert=0, axis=0, slide=0.5)
    
    #r = PrimReward(0.5, 0.5)
    #env = Environment(PrimAction.ground(), r)
    s = PrimState.initial()
    e = Experience(s, a1(s), a1, 5.0)

    r = ReplayBuffer(10)
    for _ in range(8):
        r.push(e)

    d = RBDataLoader(r, 5, batch_size=4)
    m = PrimModel(300, 650)
    m.set_reference(torch.rand(128, 128))
    b = next(iter(d))['src']
    t1 = time.time()
    m(b)
    print("Forward time: " + str(time.time() - t1))

    verts1 = torch.FloatTensor([[1,2,3],[4,5,6]])
    verts2 = torch.ones(2,3)
    prims = [
        Cuboid(verts1),
        Cuboid(verts2)
    ]
    print(prims[0].to_geom_data())
    
    PrimState.init_state_space(4, 2)
    s = PrimState(prims)  
    
    PrimAction.init_action_space(64, 2, [0.5, 1])
    a1 = PrimAction(0, vert=0, axis=0, slide=0.5)
    a2 = PrimAction(1, delete=True)
    
    r = PrimReward(0.5, 0.5)
    env = Environment(PrimAction.ground(), r)

    env.set_state(s)
    #env.set_state(env.transition(a1))
    #s1 = env.get_state()
    #print(s1.tensorize())
    #print(a2(s1).tensorize())
    #print(a1(env.get_state()).tensorize())

    e = Experience(s, a1(s), a1, 5.0)
    s1 = e.get_destination()
    s1 = a2(s1)
    print(e.get_destination().tensorize())

    r = ReplayBuffer(3)

    for i in range(4):
        r.push(e)
        print(len(r))

    d = DataLoader(r, collate_fn=collate, batch_size=2, shuffle=True)
    print(next(iter(d)))
    
    print(s)
    print(a1)
    print(e)
    print(torch.ones(7))
    
    m = PrimModel()
    t = torch.ones(3,3)
    m.set_reference(t)
    t1 = m.get_reference()
    t1[0,0] = 0
    print(t1)
    print(m.get_reference())
    
    PrimState.init_state_space(4, 2)
    print(PrimState.initial().to_geom_data())

    PrimAction.init_action_space(27, 2, [-2, -1, 1, 2])
    print(len(PrimAction.ground()))
    """