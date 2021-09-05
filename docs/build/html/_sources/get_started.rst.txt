Getting started
==================================

This section will give instruction to setup development environment and provide examples of basic usage.

Download the project
--------------------

Clone the project from git for viewing source code.

.. code-block:: shell

   git clone https://github.com/SciEcon-GameTheory/scieconlib.git
   cd scieconlib

Installation guide
------------------
Install the package using pip.

.. code-block:: shell

   python3 -m pip install scieconlib

Sample usage
------------

You can either use the package in a notebook or any python file.

.. code-block:: python

    import scieconlib.gametheory.multi_armed_bandit as bandit
    import scieconlib

    print('version: ', scieconlib.__version__)

    # create actions
    action_1 = bandit.Action.from_array([1, 2, 3, 4, 5])
    action_2 = bandit.Action.from_array([2, 4, 5, 4, 8])
    action_3 = bandit.Action.from_array([0, 1, 2, 1, 3])

    # create agent and add actions
    agent = bandit.Agent()
    agent.add_action(action_1, verbose=1)
    agent.add_action(action_2, verbose=1)
    agent.add_action(action_3, verbose=1)

    # setup the model
    model = bandit.Model(
        agent=agent,
        agent_num=10,
        epsilon=0.1,
        epochs=500
    )

    # train the model
    model.train()

    # draw the result
    model.draw_avg_freq()
    model.draw_avg_freq()

And the result will be looking like

.. figure:: ./imgs/setupResult.png

