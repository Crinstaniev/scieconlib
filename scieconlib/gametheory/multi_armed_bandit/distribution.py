import numpy as np
import plotly.figure_factory as ff


class Distribution(object):
    def __init__(self, dist_func):
        self.dist_func = dist_func

    def generate(self):
        """
        Generate random number by some distribution

        :return: random number from distribution
        :rtype: float
        """
        return self.dist_func()

    @classmethod
    def from_dist_func(cls, dist_func):
        """
        Generate a :class:`scieconlib.gametheory.multi_armed_bandit.distribution.Distribution` object from distribution
        function

        .. code-block:: python

            import numpy as np

            def dist_func():
                return np.random.normal

            distribution = Distribution.from_dist_func(dist_func)

        :param dist_func: distribution function
        :type dist_func: function
        :return: :class:`scieconlib.gametheory.multi_armed_bandit.distribution.Distribution` object
        :rtype: object
        """
        sample = dist_func()
        assert isinstance(sample, float)

        return cls(dist_func)

    def dist_plot(self, samples=1000, show=True):
        """
        Plot the distribution histogram

        :param samples: number of samples
        :type samples: int
        :param show: whether to show plot
        :type show: bool
        :return: plotly figure object
        :rtype: plotly.Figure
        """
        data = [[float(self.generate()) for _ in range(samples)]]
        fig = ff.create_distplot(data, ['sample'])
        fig.update_layout(
            title='Sample Distribution',
            xaxis_title='Sample Data'
        )
        if show:
            fig.show()
            return
        return fig

    @staticmethod
    def array_to_dist(arr):
        """
        Convert numpy array/list to distribution function

        :param arr: numpy array
        :type arr: list/ndarray
        :return: distribution function
        """
        arr = np.array(arr, dtype=np.float)

        def dist_func():
            return np.random.choice(arr, 1, replace=False)[0]

        return dist_func

    @classmethod
    def from_array(cls, arr):
        """
        Generate :class:`scieconlib.gametheory.multi_armed_bandit.distribution.Distribution` object from array

        .. code-block:: python

            action = Action.from_array([1, 2, 3, 4, 5])

        :param arr: sample array
        :type arr: list/ndarray
        :return: :class:`Distribution` object
        :rtype: scieconlib.gametheory.multi_armed_bandit.distribution.Distribution
        """
        dist_func = cls.array_to_dist(arr)
        return cls.from_dist_func(dist_func)
