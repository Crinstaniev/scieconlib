import unittest
from .agent import Agent
from .model import Model


class TestAgent(Agent):

    def roll(self, info):
        pass

    def update(self, eval_res):
        pass


class TestModel(Model):

    def rolling_info(self):
        return 0.8

    def eval(self, res):
        pass


class MyTestCase(unittest.TestCase):
    @staticmethod
    def test_agent():
        print("\n=== test agent ===")
        agent = TestAgent(5)
        # agent.table.loc[0, 'count'] = 100
        agent.get_table(verbose=1)
        return

    @staticmethod
    def test_model():
        print("\n=== test model ===")
        model = TestModel()
        agent = TestAgent(5)
        model.add_agent(agent, verbose=1)
        model.add_agent(agent, verbose=1)
        model.compile()
        model.train()
        print(model.history)

    @staticmethod
    def test_prison():
        print("=== test prisoner's dilemma ===")
        return


if __name__ == '__main__':
    unittest.main()
