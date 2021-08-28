import numpy as np
import copy
import pandas as pd

from .action import Action


class Agent(object):
    """
    This class defined an agent
    """

    def __init__(self):
        """
        Constructor method
        """
        # initial values
        self.Q = []
        self.T = []
        self.cum = []

        self.action_num = 0
        self.actions = []

    def add_action(self, action, verbose=0):
        """
        Insert an action

        .. code-block:: python

            import scieconlib.gametheory.multi_armed_bandit as bandit

            agent = Agent()
            action = bandit.Action.from_array([1, 2, 3, 4, 5])

            agent.add_action(action, verbose=1)

        :param action: the action to insert
        :type action: Action
        :param verbose: verbose=0: show nothing. verbose=1: print the action info
        :type verbose: int
        :return: None
        """
        assert isinstance(action, Action)
        action = copy.deepcopy(action)
        self.action_num += 1
        action.set_num(self.action_num - 1)
        self.actions.append(action)
        self.Q = [0 for _ in range(self.action_num)]
        self.T = [0 for _ in range(self.action_num)]
        self.cum = [0 for _ in range(self.action_num)]

        if verbose == 1:
            print(f'Added: {action.get_num()}: ', action)
        return

    def take(self, action_num):
        """
        Take an action

        :param action_num: action number to take
        :type action_num: int
        :return: None
        """
        action = self.actions[action_num]
        reward = action.generate()
        self.T[action_num] += 1
        self.cum[action_num] += reward
        self.Q[action_num] = np.divide(self.cum[action_num], self.T[action_num])
        return

    def get_avg(self):
        """
        Calculate the average value

        :return: average value
        :rtype: float
        """
        return sum(self.cum) / sum(self.T)

    def get_greedy(self):
        """
        Rank the expected values and return the number of largest

        :return: number of largest expected action
        :rtype: int
        """
        return np.argmax(np.array(self.Q))

    def pick_action(self):
        """
        Pick a random action

        :return: a random action number
        :rtype: int
        """
        num = np.random.randint(0, self.action_num)
        return num

    def get_info(self, printing=True, desc=False):
        """
        Printing actions info

        :param desc: whether to sort the dataframe
        :type desc: bool
        :param printing: whether to print
        :type printing: bool
        :return: Info dataframe
        :rtype: pd.DataFrame
        """
        print(f'Number of actions: {self.action_num}')
        df = pd.DataFrame()
        df['Action counting'] = self.T
        df['No.'] = [x for x in range(self.action_num)]
        df['Total reward'] = self.cum
        df['Expected reward'] = self.Q
        df = df.set_index('No.')
        if desc:
            df.sort_values(by='Expected reward', inplace=True, ascending=False)
        if printing:
            print(df)
            return

        return df
