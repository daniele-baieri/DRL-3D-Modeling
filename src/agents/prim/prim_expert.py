from typing import List
from agents.environment import Environment
from agents.prim.prim_state import PrimState
from agents.prim.prim_action import PrimAction
from agents.prim.prim_reward import PrimReward


class PrimExpert:

    def __init__(self, r: PrimReward, env: Environment):
        self.__reward = r
        self.__env = env

    def _poll(self, s: PrimState, max_steps: int) -> List[PrimAction]:
        res = []
        curr = s
        for step in range(max_steps):
            top = PrimAction.slide_actions if step < max_steps/2 else PrimAction.act_space_size
            #compute best action, in range (0,top)
            best, bestAct = None, None
            for act in range(top):
                A = self.__env.get_action(act)
                new = self.__reward(curr, A(curr))
                if best is None or new > best:
                    best = new
                    bestAct = A
            curr = A(curr)
            res.append(A)
        return res

    def __call__(self, s: PrimState, max_steps: int) -> PrimAction:
        return self._poll(s, max_steps)

