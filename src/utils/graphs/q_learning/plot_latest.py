

from src.config import (
    Q_LEARNING_RUN_NAME,
    DOUBLE_Q_LEARNING_RUN_NAME,
)
from src.utils.graphs.q_learning.q_learning import plot_latest_dqn_like_run


def plot_latest_dqn_run() -> list[str]:
    return plot_latest_dqn_like_run(Q_LEARNING_RUN_NAME)


def plot_latest_double_dqn_run() -> list[str]:
    return plot_latest_dqn_like_run(DOUBLE_Q_LEARNING_RUN_NAME)


if __name__ == "__main__":
    paths = plot_latest_dqn_run()

    for path in paths:
        print(path)