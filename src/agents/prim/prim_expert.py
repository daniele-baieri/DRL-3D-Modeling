from tqdm import tqdm
from typing import List, Tuple

from agents.environment import Environment
from agents.experience import Experience
from agents.imitation.expert import Expert
from agents.prim.prim_state import PrimState
from agents.prim.prim_action import PrimAction
from agents.prim.prim_reward import PrimReward

from agents.base_model import BaseModel

from trimesh import Scene


class PrimExpert(Expert):

    def __init__(self, env: Environment):
        self.__env = env

    def poll(self, s: PrimState, prim: int, delete: bool) -> Experience:
        """
        Expert policy: returns the expert's Experience on state @s.
        """
        best, bestAct = None, None
        lo = prim * PrimAction.act_per_prim
        hi = lo + PrimAction.act_per_prim - (1-int(delete))
        #if s.is_deleted(prim):
        #    hi = lo+1

        for act in range(lo, hi): 
            A = self.__env.get_action(act)
            A.set_index(act)
            new = self.__env.eval_reward(s, A(s))
            if best is None or new > best:
                best = new
                bestAct = A
        return Experience(s, bestAct(s), bestAct, best)

    def unroll(self, s: PrimState, max_steps: int) -> List[Experience]:
        """
        Runs this expert to produce an episode of length @max_steps, starting from state @s.
        """
        res = []
        curr = s
        P = PrimState.num_primitives

        iterations = tqdm(range(max_steps), desc="Expert is unrolling...")
        for step in iterations:
            prims = [i for i in range(P) if not curr.is_deleted(i)] # Primitives we can operate
            exp = self.poll(curr, prims[step%len(prims)], step > max_steps//2)
            succ = exp.get_destination()
            #assert len(succ) > 0

            res.append(exp)
            curr = succ #test that this doesn't break anything (print the experience)
            iterations.set_description("Expert unrolling - Reward: " + str(exp.get_reward().item()))
        return res

    def relabel(self, exps: List[Experience]) -> List[Experience]:
        """
        Relabels a list of experiences using the Expert's policy.
        """
        res = []
        P = PrimState.num_primitives
        max_steps = len(exps)
        iterations = tqdm(range(max_steps), desc="Expert is relabeling...")
        for step in iterations:
            curr = exps[step].get_source()
            prims = [i for i in range(P) if not curr.is_deleted(i)] # Primitives we can operate
            e_new = self.poll(curr, prims[step%len(prims)], step > max_steps//2)
            res.append(e_new)
            iterations.set_description("Expert relabeling - Reward: " + str(e_new.get_reward().item()))
        return res