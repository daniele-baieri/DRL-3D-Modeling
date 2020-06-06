from typing import List, Tuple
from agents.environment import Environment
from agents.experience import Experience
from agents.expert import Expert
from agents.prim.prim_state import PrimState
from agents.prim.prim_action import PrimAction
from agents.prim.prim_reward import PrimReward


class PrimExpert(Expert):

    def __init__(self, r: PrimReward, env: Environment):
        self.__reward = r
        self.__env = env

    def poll(self, s: PrimState, act_bound: int) -> Tuple[PrimAction, float]:
        #compute best action, in range (0,top)
        best, bestAct = None, None
        curr = s
        for act in range(act_bound):
            A = self.__env.get_action(act)
            new = self.__reward(curr, A(curr))
            if best is None or new > best:
                best = new
                bestAct = A
        return bestAct, best

    def gen_action_sequence(self, s: PrimState, max_steps: int) -> List[Experience]:
        res = []
        curr = s
        for step in range(max_steps):
            top = PrimAction.slide_actions if step < max_steps/2 else PrimAction.act_space_size
            A, r = self.poll(s, top)
            succ = A(curr)
            res.append(Experience(curr, succ, A, r))
            curr = succ #test that this doesn't break anything (print the experience)
        return res