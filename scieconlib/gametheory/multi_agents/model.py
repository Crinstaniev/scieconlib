from abc import ABC, abstractmethod

import pandas as pd
import numpy as np

from .agent import Agent
from copy import deepcopy
from tqdm import tqdm


class Model(ABC):
    """
    This class defines a model for multi-agent RL in gametheory

    :param epochs: number of training iteration
    :type epochs: int
    :param agent_copies_num: number of agent set copies you want to make
    :type agent_copies_num: int
    """

    def __init__(self, epochs: int = 1000, agent_copies_num: int = 1000):
        self.agents = []
        self.epochs = epochs
        self.agent_copies_num = agent_copies_num
        self.agents_copies = []
        self.history = []

    def add_agent(self, agent: Agent, verbose: int = 0):
        """
        Add an agent

        :param agent: target agent
        :type agent: scieconlib.gametheory.multi_agents.agent.Agent
        :param verbose: whether to print info
        :type verbose: int
        :return:
        """
        self.agents.append(deepcopy(agent))
        if verbose == 1:
            print(f'agent added: {agent}')

    @abstractmethod
    def eval(self, rolling_res: list):
        """
        Evaluate how to update the agent state by some rules.
        Should return a state for updating the agent

        :param rolling_res: list of action number chosen by agents
        :type rolling_res: list
        :return: return the instruction for how to update
        :rtype: object
        """
        raise NotImplementedError

    def compile(self):
        """
        Copy the agents by agent_num
        """
        self.agents_copies = [deepcopy(self.agents)
                              for _ in range(self.agent_copies_num)]

    @property
    @abstractmethod
    def rolling_info(self):
        """
        This method should return the extra information needed for an agent to take a roll,
        for example, epsilon value if using epsilon-greedy

        :return: some info object
        :rtype: any
        """
        return None

    def train(self, verbose: int = 1):
        """
        Train the model

        :param verbose: whether to print progress line
        :type verbose: int
        """
        assert (len(self.agents_copies) > 0)
        epoch_iter = range(self.epochs)
        if verbose:
            epoch_iter = tqdm(range(self.epochs))

        table_hist = [[] for _ in range(len(self.agents))]
        for epoch in epoch_iter:
            # epoch start
            table_avg = []
            for agent_set in self.agents_copies:
                # generate the action number set
                roll_res = []
                for agent in agent_set:
                    roll_res.append(agent.roll(self.rolling_info))
                # update agent status by model evaluation
                eval_res = self.eval(roll_res)
                table_rec = []
                for agent in agent_set:
                    agent.update(eval_res)
                    table_rec.append(agent.table.sort_index())
                table_rec = np.array(table_rec)
                table_avg.append(table_rec)
            table_avg = np.array(table_avg)
            table_avg = table_avg.sum(axis=0) / self.agent_copies_num
            for i in range(len(self.agents)):
                table_hist[i].append(table_avg[i])
            # epoch end
        history = []
        for i in range(len(table_hist)):
            payload = np.array(table_hist[i])
            history.append(payload)
        self.history = np.array(history)

    def _get_history(self, agent_num: int, action_num: int):
        """
        Retrieve history of specific agent and action number

        :param agent_num: agent number
        :param action_num: action number
        :return: history dataframe
        :rtype: pandas.DataFrame
        """
        return self.history[agent_num, :, action_num]

    def get_history(self):
        """
        Get the history table

        :return: history dataframe list
        :rtype: list[pandas.DataFrame]
        """
        history = []
        for agent in self.agents:
            hist_collect = []
            for i in range(agent.action_num):
                hist = self._get_history(agent_num=agent.number, action_num=i)
                hist_collect.append(hist)
            df = []
            for item in zip(*hist_collect):
                df.append(np.array(item).reshape(-1))
            df = pd.DataFrame(np.array(df))
            df.index.name = 'epoch'
            cols = np.array([[
                f'count_{x}',
                f'cum_{x}',
                f'avg_{x}'
            ] for x in range(agent.action_num)]).reshape(-1)
            df.columns = cols
            df.index += 1
            history.append(df)
        return history
