from scieconlib.gametheory.multi_agents.agent import Agent
from scieconlib.gametheory.multi_agents.model import Model
from copy import deepcopy

import numpy as np


class PrisonerAgent(Agent):
    """
    Implement the prisoner agent in Prisoner's Dilemma
    """

    def __init__(self):
        """
        We define two action.
        ``action 0`` for staying silence and
        ``action 1`` for betraying
        """
        super().__init__(action_num=2)

    def roll(self, epsilon):
        """
        Generate a action number.
        Determine whether to explore or exploit by epsilon value

        :param epsilon: epsilon value
        :type epsilon: float
        :return: action to take
        :rtype: int
        """
        exploit = int(self.table.index[0])
        # generate random float
        rand = np.random.uniform(0, 1, 1)[0]
        if rand <= epsilon:
            # explore
            return np.random.randint(self.action_num)
        else:
            # exploit
            return exploit

    def update(self, eval_res: tuple):
        """
        Update the table according to evaluation result

        :param eval_res: reward list of agents
        :type eval_res: list
        """
        rewards, actions = eval_res
        reward = rewards[self.number]
        action = actions[self.number]
        # update table
        self.table.loc[action, 'count'] += 1
        self.table.loc[action, 'cum'] += reward
        self.table.loc[action, 'avg'] = self.table.loc[action, 'cum'] / self.table.loc[action, 'count']
        # finally sort the table by avg
        self.table.sort_values(by='avg', inplace=True, ascending=False)


class PrisonerModel(Model):
    """
    Implement the model to train prisoner agent and we use **epsilon-greedy** algorithm
    here for stepping

    :param agent: prisoner agent class *note: it is a class, not an instance*
    :type agent: scieconlib.gametheory.multi_armed_bandit.agent.Agent
    :param epsilon: epsilon value
    :type epsilon: float
    """

    def __init__(self, agent: PrisonerAgent, epsilon: float = 0.1, epochs: int = 1000, agent_copies_num: int = 1000):
        super().__init__(
            epochs=epochs,
            agent_copies_num=agent_copies_num
        )
        self.epsilon = epsilon
        self.add_agent(deepcopy(agent.set_number(0)), verbose=1)
        self.add_agent(deepcopy(agent.set_number(1)), verbose=1)

    @property
    def rolling_info(self):
        """
        Return the epsilon value for rolling

        :return: epsilon value
        :rtype: float
        """
        epsilon = 0.1
        return epsilon

    def eval(self, rolling_res: list):
        """
        Evaluate the action set by rule of prisoner's dilemma.
        We define ``action 0`` for staying silent
        and ``action 1`` for betraying.
        We have two agents ``A`` and ``B``.
        If both stay silent, both ``A`` and ``B`` will get ``-1`` reward.
        If both betrays, both ``A`` and ``B`` will get ``-2`` reward.
        If one stay silent and one betray, the betrayer get ``0`` reward
        and the another get ``-3`` reward.

        **Reference** https://en.wikipedia.org/wiki/Prisoner%27s_dilemma

        :param rolling_res: the actions chosen by agents
        :type rolling_res: list
        :return: tuple of evaluation and action list
        :rtype: tuple[list[float], list[int]]
        """
        rewards = None
        a = rolling_res[0]
        b = rolling_res[1]
        if a == 0 and b == 0:
            # both stay silent
            rewards = [-1, -1]
        elif a == 1 and b == 0:
            # A betray
            rewards = [0, -3]
        elif a == 0 and b == 1:
            # B betray
            rewards = [-3, 0]
        elif a == 1 and b == 1:
            # both betray
            rewards = [-2, -2]
        return rewards, rolling_res
