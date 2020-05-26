import torch
from agents.prim.prim_state import PrimState
from agents.prim.prim_action import PrimAction
from agents.environment import Environment
from agents.experience import Experience
from agents.replay_buffer import ReplayBuffer
from geometry.cuboid import Cuboid


def test():
    verts1 = torch.FloatTensor([[0,1,0],[1,0,0]])
    verts2 = torch.ones(2,3)
    prims = [
        Cuboid(verts1),
        Cuboid(verts2)
    ]
    
    PrimState.init_state_space(2)
    s = PrimState(prims)  
    
    PrimAction.init_action_space(2, 2)
    a1 = PrimAction(0, vert=0, axis=0, slide=0.5)
    a2 = PrimAction(1, delete=True)
    
    env = Environment()

    env.set_state(s)
    env.set_state(env.transition(a1))
    s1 = env.get_state()
    print(s1.tensorize())
    print(a2(s1).tensorize())
    print(a1(env.get_state()).tensorize())

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


if __name__ == "__main__":
    test()


