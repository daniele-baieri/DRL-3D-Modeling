from tqdm import tqdm
from typing import List, Tuple
from agents.environment import Environment
from agents.experience import Experience
from agents.expert import Expert
from agents.prim.prim_state import PrimState
from agents.prim.prim_action import PrimAction
from agents.prim.prim_reward import PrimReward

from agents.base_model import BaseModel

from trimesh import Scene


class PrimExpert(Expert):

    def __init__(self, r: PrimReward, env: Environment):
        self.__reward = r
        self.__env = env

    def poll(self, s: PrimState, prim: int, delete: bool) -> Tuple[PrimAction, float]:
        best, bestAct = None, None
        curr = s
        lo = prim * PrimAction.act_per_prim
        hi = lo + PrimAction.act_per_prim - (1-int(delete))
        for act in range(lo, hi): 
            A = self.__env.get_action(act)
            #print(A)
            new = self.__reward(curr, A(curr))
            if best is None or new > best:
                best = new
                bestAct = A
        #print(best)
        return bestAct, best

    def get_action_sequence(self, s: PrimState, max_steps: int) -> List[Experience]:
        res = []
        curr = s
        step = 0
        while step < max_steps:
            for prim in tqdm(range(PrimState.num_primitives)):
                step += 1
                delete = step > max_steps//2 
                A, r = self.poll(curr, prim, delete)
                succ = A(curr)
                assert len(succ) > 0

                #scene = Scene()
                #scene.add_geometry(succ.meshify())
                #scene.add_geometry(BaseModel.model)
                #scene.show()

                res.append(Experience(curr, succ, A, r))
                curr = succ #test that this doesn't break anything (print the experience)
        return res