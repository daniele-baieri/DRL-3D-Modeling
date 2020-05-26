from agents.state import State


class Action:

    def _apply(self, s: State) -> State:
        raise NotImplementedError

    def __call__(self, s: State) -> State:
        return self._apply(s)