import time, math, os
import torch
import trimesh

from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam

#from torch_geometric.utils import to_trimesh

from agents.prim.prim_state import PrimState
from agents.prim.prim_action import PrimAction
from agents.prim.prim_model import PrimModel
from agents.prim.prim_reward import PrimReward
from agents.prim.prim_expert import PrimExpert
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
'''

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
    target = PrimModel(PrimAction.act_space_size)
    target.load_state_dict(online.state_dict())
    target.eval()
    
    opt = Adam(online.parameters(), 0.00008)
    exp = PrimExpert(env)

    trainer = DoubleDQNTrainer(online, target, env, opt, exp, 0.9, 0.8, 20, 0.02, device)

    rfc = ShapeDataset('../data/ShapeNet', items_per_category={'plane': 800, 'pistol': 600, 'rocket': 800})
    imit = ShapeDataset('../data/ShapeNet', items_per_category={'plane': 800, 'pistol': 600, 'rocket': 800}, partition='IMIT')
    print("IMITATION DATA: " + str(len(imit)) + " instances")
    print("REINFORCEMENT DATA: " + str(len(rfc)) + " instances")
    t1 = time.time()
    trainer.train(rfc, imit, PrimState.episode_len, 200000, 100000, 64, 4, 500, '../model/PRIM.pth')
    print("Training time: " + str(time.time() - t1))

    
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

    #os.environ['DEVICE'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    t = time.time()
    #virtual_expert_modeling()
    test()
    print("Test time: " + str(time.time() - t))

    # old tests

    '''
    print("Begin")
    PrimState.init_state_space(3, 64)
    PrimAction.init_action_space(PrimState.num_primitives, 2, [-0.5 , 1.0])
    s = PrimState.initial()
    print("Initialized")
    a1 = PrimAction(0, vert=0, axis=1, slide=-0.5)
    #a2 = PrimAction(1, vert=0, axis=0, slide=0.5)
    #a3 = PrimAction(2, vert=1, axis=1, slide=0.3)
    #a4 = PrimAction(8, delete=True)
    #a5 = PrimAction(0, vert=0, axis=1, slide=0.3)
    s1 = a1(s)
    s1.meshify().show()
    print("Computed successor")
    x = s1.voxelize()

    D = ShapeDataset('/home/bayo/Documents/CS1920/DLAI-s2-2020/project/data/ShapeNet', categories=['knife'])
    M = next(iter(D))
    R = PrimReward(0.2, 0.01)
    BaseModel.new_episode(M['reference'], M['mesh'])

    t = time.time()
    print(R(s, s1))
    print(time.time() - t)
    '''
    '''
    #print(M.shape)
    V = s.voxelize(cubes=True)
    print(torch.unique(torch.cat(V), sorted=False).shape)

    BaseModel.new_episode(ref, M)

    s1 = s
    actions = [PrimAction(i, delete=True) for i in range(10)]
    for act in actions:
        s1 = act

    #m = s.meshify()
    R = PrimReward(0.2, 0.01)
    t = time.time()
    print(R(s, s1))
    print("Reward time: " + str(time.time() - t))
    '''
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