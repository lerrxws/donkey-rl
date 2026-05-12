# Donkey RL

This project trains reinforcement learning agents to play the DOS Donkey game through DOSBox. The game is treated as an external environment: the agent observes the screen, detects the player car and donkey with template matching, builds a compact state vector, selects an action, and receives rewards based on score events and driving behavior.

The repository contains implementations and experiment tooling for:

- DQN-style Q-learning
- Double DQN
- One-step Actor-Critic
- Episodic Actor-Critic

## Requirements

The project is designed for a Windows setup with DOSBox installed. The default path in [src/config.py](src/config.py) is:

```text
C:\Program Files (x86)\DOSBox-0.74-3\DOSBox.exe
```

Required local assets:

- `donkey/DONKEY.EXE`
- `dosbox.conf`
- object and score templates under `data/templates/`

Python dependencies are listed in [requirements.txt](requirements.txt):

- `opencv-python`
- `mss`
- `pygetwindow`
- `pyautogui`
- `numpy`
- `torch`
- `matplotlib`
- `pandas`

Set up the Python environment:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## How It Works

Training starts from [main.py](main.py), which calls `run_training` from [src/training.py](src/training.py). Each run follows this process:

1. Start DOSBox with the project `dosbox.conf`.
2. Find and activate the DOSBox window.
3. Capture the game window.
4. Detect the player car, donkey, and score counters from the frame.
5. Convert detections into a compact state vector.
6. Let the selected agent choose an action.
7. Apply the action to the game.
8. Compute reward from score changes and state/action context.
9. Log step and episode metrics.
10. Save CSV files, checkpoints, and training graphs.

The simple state vector used by the agents is:

```text
[car_line, donkey_line, danger]
```

where:

- `car_line` is the detected lane of the player car.
- `donkey_line` is the detected lane of the donkey.
- `danger` is a binary feature set to `1` when the donkey is visible, in the same lane, and inside the configured vertical danger interval.

The raw detection state contains normalized object coordinates and visibility flags. The final state is built in [src/env/state_builder.py](src/env/state_builder.py).

The action space has two actions:

```text
0 = no jump
1 = jump/change
```

## Reinforcement Learning Methods

### DQN

The DQN agent approximates the action-value function \(Q(s,a)\) with a neural network. It uses epsilon-greedy exploration, a replay buffer, and a target network. The target network stabilizes learning by providing a slower-moving estimate for bootstrapping.

### Double DQN

Double DQN uses the training network to select the next action and the target network to evaluate that selected action. This reduces the overestimation bias that can occur when the same network both selects and evaluates the maximum next-state action.

### Actor-Critic

Actor-Critic methods combine a policy model and a value model. The actor learns the policy \(\pi(a|s)\), while the critic estimates the state value \(V(s)\). In the one-step version, each transition is updated with the temporal-difference error:

```text
delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
```

The critic uses this target to improve value estimates, and the actor uses the TD error as an advantage signal for improving the policy.

### Reward Shaping

Base rewards come from game score events:

- crash: negative terminal reward
- lap/score increase: positive reward
- normal survival step: small positive reward

Additional shaping rewards and penalties are applied based on the detected state:

- penalize jumping when the donkey is visible but not in the same lane
- penalize unnecessary jumps outside danger
- penalize not jumping when danger is detected
- reward useful jumps in danger

This shaping is implemented in [src/env/reward.py](src/env/reward.py).

## Running Training

Run the current experiment script:

```bash
python main.py
```

The current [main.py](main.py) runs DQN and Double DQN experiments over several random seeds and hidden-layer configurations. Experiment settings are split between:

- [src/config.py](src/config.py): paths, state/action sizes, rewards, detection thresholds, and maximum episode steps
- [main.py](main.py): selected modes, random seeds, hidden layers, and run loop

To run a different algorithm or configuration, edit the calls to `run_training` in [main.py](main.py).

## Outputs

Training runs are saved under:

```text
data/runs/<algorithm>_<timestamp>/
```

Each run can contain:

- `config.json`: run configuration
- `episodes.csv`: episode-level metrics
- `steps.csv`: step-level metrics
- `graphs/`: generated plots for rewards, losses, TD error, entropy, and value estimates

Model checkpoints are saved under:

```text
checkpoints/
```

Analysis and plotting helpers are stored in:

```text
scripts/
```

These scripts are used to aggregate runs, generate comparison graphs, and create LaTeX tables for reports. The repository ignores `data/*` by default, so generated run data is not automatically tracked. Selected experiment runs can be added manually when they are needed for reproducibility or reporting.

## Project Layout

```text
src/agents/      RL agents and neural networks
src/detection/   screen, object, and score detection
src/env/         episode loop, reward logic, and state builder
src/utils/       logging, metrics, graphing, seeds, and capture helpers
data/runs/       training outputs
images/          generated thesis/report figures
scripts/         analysis and plotting helpers
donkey/          DOS game executable
```

## Notes

This project controls a real DOSBox window through screen capture and keyboard input. Keep the DOSBox window visible and avoid moving or covering it during training, otherwise object detection and action timing can become unreliable.
