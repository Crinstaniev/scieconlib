import pandas as pd
from abc import ABC, abstractmethod


class Agent(ABC):
    """
    This class define an agent class for dual agent RL in game theory.

    :param action_num: number of actions
    :type action_num: int
    """

    def __init__(self, action_num: int):
        # initialize table
        zeros = [0 for _ in range(action_num)]
        table = pd.DataFrame({
            'number': [x for x in range(action_num)],
            'count': zeros,
            'cum': zeros,
            'avg': zeros
        })
        table['number'] = table['number'].astype(int)
        table['count'] = table['count'].astype(int)
        table['cum'] = table['count'].astype(float)
        table['avg'] = table['avg'].astype(float)
        table.set_index('number', inplace=True)
        self.action_num = action_num
        self.table = table
        self.actions = []
        self.number = None

    def get_table(self, verbose: int = 0):
        """
        Get the table

        :param verbose: whether to print info
        :type verbose: int
        :return: value table
        :rtype: pandas.DataFrame
        """
        if verbose == 1:
            print(self.table)
        return self.table

    @abstractmethod
    def roll(self, info):
        """
        This method generate a action number according to some rule

        :param info: information needed for doing a roll
        :type info: any
        :return: action number
        :rtype: int
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, eval_res: object):
        """
        This method update the table by some rule

        :param eval_res: model evaluation result
        :type eval_res: object
        """
        raise NotImplementedError

    def set_number(self, number: int):
        """
        Set the agent number

        :param number: agent number
        :type number: int
        :return: the agent object
        :rtype: scieconlib.gametheory.multi_agents.agent.Agent
        """
        self.number = number
        return self
