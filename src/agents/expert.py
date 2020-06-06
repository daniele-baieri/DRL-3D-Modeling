from typing import Tuple, List

from agents.action import Action
from agents.state import State
from agents.experience import Experience


class Expert:

    def poll(self, s: State) -> Action:
        raise NotImplementedError

    def get_action_sequence(self, s: State, max_steps: int) -> List[Experience]:
        raise NotImplementedError