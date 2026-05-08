import os

import numpy as np

from src.utils.graphs.base import (
    ensure_graph_dir,
    has_columns,
    numeric_column,
    read_csv_rows,
    save_line_plot,
    save_line_with_band,
)


def load_actor_critic_episodes(run_dir: str) -> tuple[list[dict[str, str]], np.ndarray, str]:
    episodes_path = os.path.join(run_dir, "episodes.csv")
    rows = read_csv_rows(episodes_path)
    graph_dir = ensure_graph_dir(run_dir)

    if not has_columns(rows, ["episode"]):
        return rows, np.asarray([], dtype=np.float32), graph_dir

    return rows, numeric_column(rows, "episode"), graph_dir


def plot_total_reward(
    rows: list[dict[str, str]],
    x: np.ndarray,
    graph_dir: str,
) -> str | None:
    if not has_columns(rows, ["episode", "total_reward"]):
        return None

    return save_line_plot(
        x=x,
        series={"total_reward": numeric_column(rows, "total_reward")},
        title="Total reward per episode",
        ylabel="total_reward",
        path=os.path.join(graph_dir, "total_reward.png"),
    )


def plot_episode_steps(
    rows: list[dict[str, str]],
    x: np.ndarray,
    graph_dir: str,
) -> str | None:
    if not has_columns(rows, ["episode", "episode_steps"]):
        return None

    return save_line_plot(
        x=x,
        series={"episode_steps": numeric_column(rows, "episode_steps")},
        title="Episode length",
        ylabel="episode_steps",
        path=os.path.join(graph_dir, "episode_steps.png"),
    )


def plot_entropy(
    rows: list[dict[str, str]],
    x: np.ndarray,
    graph_dir: str,
) -> str | None:
    if not has_columns(rows, ["episode", "entropy_mean_mean"]):
        return None

    mean = numeric_column(rows, "entropy_mean_mean")
    path = os.path.join(graph_dir, "entropy.png")

    if has_columns(rows, ["entropy_mean_std"]):
        std = numeric_column(rows, "entropy_mean_std")
        return save_line_with_band(
            x=x,
            mean=mean,
            low=mean - std,
            high=mean + std,
            title="Policy entropy",
            ylabel="entropy",
            path=path,
            label="entropy_mean",
        )

    return save_line_plot(
        x=x,
        series={"entropy_mean": mean},
        title="Policy entropy",
        ylabel="entropy",
        path=path,
    )


def plot_value(
    rows: list[dict[str, str]],
    x: np.ndarray,
    graph_dir: str,
) -> str | None:
    if not has_columns(rows, ["episode", "value_mean"]):
        return None

    path = os.path.join(graph_dir, "value.png")
    mean = numeric_column(rows, "value_mean")

    if has_columns(rows, ["value_min", "value_max"]):
        return save_line_with_band(
            x=x,
            mean=mean,
            low=numeric_column(rows, "value_min"),
            high=numeric_column(rows, "value_max"),
            title="Value estimate",
            ylabel="value",
            path=path,
            label="value_mean",
        )

    return save_line_plot(
        x=x,
        series={"value_mean": mean},
        title="Value estimate",
        ylabel="value",
        path=path,
    )


def plot_actor_critic_common_graphs(run_dir: str) -> list[str]:
    rows, x, graph_dir = load_actor_critic_episodes(run_dir)
    paths = []

    for plotter in (
        plot_total_reward,
        plot_episode_steps,
        plot_entropy,
        plot_value,
    ):
        path = plotter(rows, x, graph_dir)

        if path is not None:
            paths.append(path)

    return paths
