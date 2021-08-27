from .distribution import Distribution


class Action(object):
    """
    This class defined an action.
    :param distribution: The distribution function
    :type distribution: Distribution
    """

    def __init__(self, distribution):
        """
        Constructor method
        """
        # make sure the type
        assert isinstance(distribution, Distribution)

        self.distribution = distribution
        self.number = 0

    @classmethod
    def from_distribution_func(cls, dist_func):
        """
        Generate an :class:`Action` object from distribution function

        .. code-block:: python

            import numpy as np

            def dist_func():
                return np.random.normal

            action = Action.from_distribution_func(dist_func)

        :param dist_func: function that returns a random number from the distribution
        :type dist_func: function
        :return: :class:`Action` object
        :rtype: Action
        """
        dist = Distribution.from_dist_func(dist_func)
        return cls(dist)

    @classmethod
    def from_array(cls, arr):
        """
        Construct an :class:`Action` from samples

        .. code-block:: python

            action = Action.from_array([1, 2, 3, 4, 5])

        :param arr: sample array
        :type arr: list
        :return: :class:`Action` object
        :rtype: Action
        """
        distribution = Distribution.from_array(arr)
        return cls(distribution)

    def generate(self):
        """
        Generate random number from distribution

        :return: random number
        :rtype: float
        """
        return self.distribution.generate()

    def get_num(self):
        """
        Get the number of action

        :return: action number
        :rtype: int
        """
        return self.number

    def set_num(self, number):
        """
        Set the action number

        :param number: target action number
        :type number: int
        :return: None
        """
        self.number = number
        return

    def hist(self):
        """
        Draw the distribution histogram

        :return: None
        """
        self.distribution.hist()
        return
