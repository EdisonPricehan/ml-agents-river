from gymnasium import spaces
from typing import List, Dict, Tuple, Union
import itertools


class ActionFlattener:
    """
    Flattens branched discrete action spaces into single-branch discrete action spaces.
    """

    def __init__(self, branched_action_space):
        """
        Initialize the flattener.
        :param branched_action_space: A List containing the sizes of each branch of the action
        space, e.g. [2,3,3] for three branches with size 2, 3, and 3 respectively.
        """
        self._action_shape = branched_action_space
        self.action_lookup = self._create_lookup(self._action_shape)
        self.action_space = spaces.Discrete(len(self.action_lookup))

    @classmethod
    def _create_lookup(self, branched_action_space):
        """
        Creates a Dict that maps discrete actions (scalars) to branched actions (lists).
        Each key in the Dict maps to one unique set of branched actions, and each value
        contains the List of branched actions.
        """
        possible_vals = [range(_num) for _num in branched_action_space]
        all_actions = [list(_action) for _action in itertools.product(*possible_vals)]
        # Dict should be faster than List for large action spaces
        action_lookup = {
            _scalar: _action for (_scalar, _action) in enumerate(all_actions)
        }
        return action_lookup

    def lookup_action(self, action):
        """
        Convert a scalar discrete action into a unique set of branched actions.
        :param action: A scalar value representing one of the discrete actions.
        :returns: The List containing the branched actions.
        """
        return self.action_lookup[action]


class DroneActionFlattener:
    def __init__(self, branched_action_space: Union[List[int], Tuple[int, ...]]):
        """
        Flatten (Convert) the "multi-discrete" action to discrete action assuming:
        1. all action branches share the same "no operation" action
        2. only 1 branch action will be active (want action on a single branch to be executed at a time, i.e., discrete)
        :param branched_action_space:
        """
        self._action_shape = branched_action_space
        self._action_size = len(branched_action_space)
        self.action_size = sum([act - 1 for act in self._action_shape]) + 1
        self.action_space = spaces.Discrete(self.action_size)
        self.action_lookup = self._create_lookup()

    def _create_lookup(self) -> Dict[int, List[int]]:
        action_lookup = {}
        for i in range(self.action_size):
            multi_discrete_action = [1] * self._action_size
            if i != 0:
                idx = (i - 1) // 2
                val = 0 if i % 2 == 1 else 2
                multi_discrete_action[idx] = val
            action_lookup[i] = multi_discrete_action
        return action_lookup

    def lookup_action(self, action: int) -> List[int]:
        return self.action_lookup[action]


if __name__ == '__main__':
    # test drone action flattener
    flattener = DroneActionFlattener([3, 3, 3, 3])
    for a in range(9):
        multi_discrete_action = flattener.lookup_action(a)
        print(f'Discrete action: {a}, multi-discrete action: {multi_discrete_action}')

