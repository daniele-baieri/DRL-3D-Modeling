import torch
from agents.prim.prim_state import PrimState
from agents.prim.prim_action import PrimAction
from geometry.cuboid import Cuboid



if __name__ == "__main__":

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

    print(a1(s).tensorize())
    print(a2(s).tensorize())
    print(a2(a1(s)).tensorize())