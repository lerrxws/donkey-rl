# src/utils/graphs/dqn/dqn.py

import os
from pathlib import Path

from src.config import RUNS_DIR
from src.utils.graphs.base import (
    ensure_graph_dir,
    has_columns,
    numeric_column,
    read_csv_rows,
    save_line_plot,
    save_line_with_band,
)


def load_dqn_episodes(run_dir: str):
    episodes_path = os.path.join(run_dir, "episodes.csv")

    if not os.path.exists(episodes_path):
        raise FileNotFoundError(f"episodes.csv not found: {episodes_path}")

    rows = read_csv_rows(episodes_path)

    if not rows:
        raise ValueError(f"episodes.csv is empty: {episodes_path}")

    graph_dir = ensure_graph_dir(run_dir)
    x = numeric_column(rows, "episode")

    return rows, x, graph_dir


def plot_total_reward(rows, x, graph_dir: str) -> str | None:
    if not has_columns(rows, ["episode", "total_reward"]):
        return None

    series = {
        "total_reward": numeric_column(rows, "total_reward"),
    }

    if has_columns(rows, ["avg_reward_10"]):
        series["avg_reward_10"] = numeric_column(rows, "avg_reward_10")

    if has_columns(rows, ["avg_reward_50"]):
        series["avg_reward_50"] = numeric_column(rows, "avg_reward_50")

    return save_line_plot(
        x=x,
        series=series,
        title="Total reward per episode",
        ylabel="reward",
        path=os.path.join(graph_dir, "total_reward.png"),
    )


def plot_episode_steps(rows, x, graph_dir: str) -> str | None:
    if not has_columns(rows, ["episode", "episode_steps"]):
        return None

    return save_line_plot(
        x=x,
        series={
            "episode_steps": numeric_column(rows, "episode_steps"),
        },
        title="Episode length",
        ylabel="steps",
        path=os.path.join(graph_dir, "episode_steps.png"),
    )


def plot_lap_count(rows, x, graph_dir: str) -> str | None:
    if not has_columns(rows, ["episode", "lap_count"]):
        return None

    return save_line_plot(
        x=x,
        series={
            "lap_count": numeric_column(rows, "lap_count"),
        },
        title="Laps per episode",
        ylabel="laps",
        path=os.path.join(graph_dir, "lap_count.png"),
    )


def plot_crash_rate(rows, x, graph_dir: str) -> str | None:
    if not has_columns(rows, ["episode", "crash_rate_50"]):
        return None

    return save_line_plot(
        x=x,
        series={
            "crash_rate_50": numeric_column(rows, "crash_rate_50"),
        },
        title="Crash rate over last 50 episodes",
        ylabel="crash rate",
        path=os.path.join(graph_dir, "crash_rate_50.png"),
    )


def plot_epsilon(rows, x, graph_dir: str) -> str | None:
    if not has_columns(rows, ["episode", "epsilon"]):
        return None

    return save_line_plot(
        x=x,
        series={
            "epsilon": numeric_column(rows, "epsilon"),
        },
        title="Epsilon decay",
        ylabel="epsilon",
        path=os.path.join(graph_dir, "epsilon.png"),
    )


def plot_mean_loss(rows, x, graph_dir: str) -> str | None:
    if not has_columns(rows, ["episode", "mean_loss"]):
        return None

    return save_line_plot(
        x=x,
        series={
            "mean_loss": numeric_column(rows, "mean_loss"),
        },
        title="Mean loss per episode",
        ylabel="loss",
        path=os.path.join(graph_dir, "mean_loss.png"),
    )


def plot_loss_range(rows, x, graph_dir: str) -> str | None:
    columns = [
        "episode",
        "loss_mean",
        "loss_min",
        "loss_max",
    ]

    if not has_columns(rows, columns):
        return None

    return save_line_with_band(
        x=x,
        mean=numeric_column(rows, "loss_mean"),
        low=numeric_column(rows, "loss_min"),
        high=numeric_column(rows, "loss_max"),
        title="Step loss range per episode",
        ylabel="loss",
        path=os.path.join(graph_dir, "loss_range.png"),
        label="loss_mean",
    )


def plot_reward_range(rows, x, graph_dir: str) -> str | None:
    columns = [
        "episode",
        "reward_mean",
        "reward_min",
        "reward_max",
    ]

    if not has_columns(rows, columns):
        return None

    return save_line_with_band(
        x=x,
        mean=numeric_column(rows, "reward_mean"),
        low=numeric_column(rows, "reward_min"),
        high=numeric_column(rows, "reward_max"),
        title="Step reward range per episode",
        ylabel="reward",
        path=os.path.join(graph_dir, "reward_range.png"),
        label="reward_mean",
    )


def plot_jump_rate(rows, x, graph_dir: str) -> str | None:
    """
    action_mean works because actions are:
        0 = no jump
        1 = jump

    So mean(action) = jump rate.
    """

    if not has_columns(rows, ["episode", "action_mean"]):
        return None

    return save_line_plot(
        x=x,
        series={
            "jump_rate": numeric_column(rows, "action_mean"),
        },
        title="Jump rate per episode",
        ylabel="jump rate",
        path=os.path.join(graph_dir, "jump_rate.png"),
    )


def plot_danger_rate(rows, x, graph_dir: str) -> str | None:
    if not has_columns(rows, ["episode", "danger_mean"]):
        return None

    return save_line_plot(
        x=x,
        series={
            "danger_rate": numeric_column(rows, "danger_mean"),
        },
        title="Danger state rate per episode",
        ylabel="danger rate",
        path=os.path.join(graph_dir, "danger_rate.png"),
    )


def plot_same_line_rate(rows, x, graph_dir: str) -> str | None:
    if not has_columns(rows, ["episode", "same_line_mean"]):
        return None

    return save_line_plot(
        x=x,
        series={
            "same_line_rate": numeric_column(rows, "same_line_mean"),
        },
        title="Same lane rate per episode",
        ylabel="same lane rate",
        path=os.path.join(graph_dir, "same_line_rate.png"),
    )


def plot_dqn_run(run_dir: str) -> list[str]:
    rows, x, graph_dir = load_dqn_episodes(run_dir)

    paths = []

    for plotter in (
        plot_total_reward,
        plot_episode_steps,
        plot_lap_count,
        plot_crash_rate,
        plot_epsilon,
        plot_mean_loss,
        plot_loss_range,
        plot_reward_range,
        plot_jump_rate,
        plot_danger_rate,
        plot_same_line_rate,
    ):
        path = plotter(rows, x, graph_dir)

        if path is not None:
            paths.append(path)

    return paths


def plot_latest_dqn_like_run(
    run_name: str,
    root_dir: str = RUNS_DIR,
) -> list[str]:
    run_root = Path(root_dir)

    candidates = sorted(
        [
            path
            for path in run_root.glob(f"{run_name}_*")
            if path.is_dir() and (path / "episodes.csv").exists()
        ],
        key=lambda path: path.stat().st_mtime,
    )

    if not candidates:
        raise FileNotFoundError(
            f"No run '{run_name}_*' with episodes.csv found in {root_dir}"
        )

    return plot_dqn_run(str(candidates[-1]))