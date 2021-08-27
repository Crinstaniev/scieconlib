import copy
import numpy as np
import plotly.express as px
import pandas as pd
from .agent import Agent
from tqdm import tqdm


class Model(object):
    """
    This class defined a model for multi-armed bandit problem

    .. code-block:: python

        import scieconlib.gametheory.multi_armed_bandit as bandit

        action_1 = bandit.Action.from_array([1, 2, 3])
        action_2 = bandit.Action.from_array([2, 3, 4])

        agent = bandit.Agent()
        agent.add_action(action_1)
        agent.add_action(action_2)

        model = Model(
            agent=agent
            agent_num=1000
            epochs=1000
            epsilon=0.15
        )

    :param agent: Agent object
    :type agent: Agent
    :type agent: Agent
    :param agent_num: number of agent to generate
    :param epsilon: primary parameter for epsilon-greedy algorithm
    :param epochs: epochs to train the model
    """

    def __init__(self, agent, agent_num=200, epsilon=0.1, epochs=1000):
        """
        Constructor method
        """
        self.agent = agent
        self.agent_num = agent_num
        self.epsilon = epsilon
        self.epochs = epochs
        self.history = None
        pass

    def _roll(self):
        """
        Roll a dice based on epsilon value

        :return: whether go greedy
        :rtype: bool
        """
        random_num = np.random.uniform(0, 1)
        if random_num <= self.epsilon:
            return True
        else:
            # do exploit
            return False

    def train(self):
        """
        train the model
        """
        # generate agents
        history = {
            'avg_reward': [],
            'num': []
        }
        cnt = 0
        agents = [copy.deepcopy(self.agent) for _ in range(self.agent_num)]
        for _ in tqdm(range(self.epochs)):
            for agent in agents:
                explore = self._roll()
                if explore:
                    action_num = agent.pick_action()
                    agent.take(action_num)
                else:
                    action_num = agent.get_greedy()
                    agent.take(action_num)
            cnt += 1
            history['num'].append(cnt)
            avg_reward = np.average([agent.get_avg() for agent in agents])
            history['avg_reward'].append(avg_reward)
        self.history = history
        return

    def history(self):
        """
        Draw the learning curve
        """
        df = pd.DataFrame(self.history)
        fig = px.line(df, x='num', y='avg_reward')
        fig.update_layout(
            title='Average Reward Curve',
            xaxis_title='epochs',
            yaxis_title='avg reward'
        )
        fig.show()
        return
