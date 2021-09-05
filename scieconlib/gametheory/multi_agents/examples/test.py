import unittest
import matplotlib.pyplot as plt
from scieconlib.gametheory.multi_agents.examples.prisoner_dilemma import PrisonerModel, PrisonerAgent


class MyTestCase(unittest.TestCase):
    @staticmethod
    def test_model():
        model = PrisonerModel(
            agent=PrisonerAgent(),
            agent_copies_num=20,
            epochs=200,
            epsilon=0.1
        )
        model.compile()
        model.train(verbose=1)
        history = model.get_history()
        fig, axs = plt.subplots(2)
        for i in range(2):
            axs[i].plot(history[i].index, history[i]['avg_0'], label='keep silence')
            axs[i].plot(history[i].index, history[i]['avg_1'], label='betray')
            axs[i].set_title(f'Agent {i} - avg value')
            axs[i].legend()
            axs[i].grid()
            axs[i].set_xlabel('epochs')
            axs[i].set_ylabel('avg-value')
        fig.set_dpi(200)
        fig.subplots_adjust(
            hspace=.5
        )
        fig.show()


if __name__ == '__main__':
    unittest.main()
