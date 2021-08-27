# scieconlib

This is a machine learning toolkit to game theory or econometrics analysis.

## Dev environment setup

In your virtual environment, run

```shell
python3 -m pip install -r requirements.txt
```

To build the project, run

```shell
make clean && make start
```

## Basic usage

### Installation

```sheel
python3 -m pip install scieconlib
```

### Example

```python
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
model.history()
```
