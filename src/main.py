import torch
from agents.prim.prim_state import PrimState
from agents.prim.prim_action import PrimAction
from agents.prim.prim_model import PrimModel
from agents.prim.prim_reward import PrimReward
from agents.environment import Environment
from agents.experience import Experience
from agents.replay_buffer import ReplayBuffer
from geometry.cuboid import Cuboid


def test():
    
    """
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
    
    
if __name__ == "__main__":
    test()


