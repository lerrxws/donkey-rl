import os
from pathlib import Path

from src.constants import ONE_STEP_ACTOR_CRITIC_RUN_NAME, RUNS_DIR
from src.utils.graphs.actor_critic.actor_critic import (
    load_actor_critic_episodes,
    plot_episode_steps,
    plot_entropy,
    plot_total_reward,
)
from src.utils.graphs.base import (
    has_columns,
    numeric_column,
    save_line_plot,
    save_line_with_band,
)


def _plot_range_metric(
    rows: list[dict[str, str]],
    x,
    graph_dir: str,
    metric: str,
    title: str,
    ylabel: str,
    output_name: str,
) -> str | None:
    columns = [
        "episode",
        f"{metric}_mean",
        f"{metric}_min",
        f"{metric}_max",
    ]

    if not has_columns(rows, columns):
        return None

    return save_line_with_band(
        x=x,
        mean=numeric_column(rows, f"{metric}_mean"),
        low=numeric_column(rows, f"{metric}_min"),
        high=numeric_column(rows, f"{metric}_max"),
        title=title,
        ylabel=ylabel,
        path=os.path.join(graph_dir, output_name),
        label=f"{metric}_mean",
    )


def _plot_td_error(
    rows: list[dict[str, str]],
    x,
    graph_dir: str,
) -> str | None:
    columns = [
        "episode",
        "td_error_abs_mean",
        "td_error_min",
        "td_error_max",
    ]

    if not has_columns(rows, columns):
        return None

    return save_line_with_band(
        x=x,
        mean=numeric_column(rows, "td_error_abs_mean"),
        low=numeric_column(rows, "td_error_min"),
        high=numeric_column(rows, "td_error_max"),
        title="TD error",
        ylabel="td_error",
        path=os.path.join(graph_dir, "td_error.png"),
        label="td_error_abs_mean",
    )


def _plot_value_vs_target(
    rows: list[dict[str, str]],
    x,
    graph_dir: str,
) -> str | None:
    if not has_columns(rows, ["episode", "value_mean", "target_mean"]):
        return None

    series = {
        "value_mean": numeric_column(rows, "value_mean"),
        "target_mean": numeric_column(rows, "target_mean"),
    }

    if has_columns(rows, ["next_value_mean"]):
        series["next_value_mean"] = numeric_column(rows, "next_value_mean")

    return save_line_plot(
        x=x,
        series=series,
        title="Value vs target",
        ylabel="value",
        path=os.path.join(graph_dir, "value_vs_target.png"),
    )


def plot_one_step_actor_critic_run(run_dir: str) -> list[str]:
    rows, x, graph_dir = load_actor_critic_episodes(run_dir)
    paths = []

    for plotter in (
        plot_total_reward,
        plot_episode_steps,
        lambda r, episodes, output_dir: _plot_range_metric(
            rows=r,
            x=episodes,
            graph_dir=output_dir,
            metric="actor_loss",
            title="Actor loss",
            ylabel="actor_loss",
            output_name="actor_loss.png",
        ),
        lambda r, episodes, output_dir: _plot_range_metric(
            rows=r,
            x=episodes,
            graph_dir=output_dir,
            metric="critic_loss",
            title="Critic loss",
            ylabel="critic_loss",
            output_name="critic_loss.png",
        ),
        _plot_td_error,
        _plot_value_vs_target,
        plot_entropy,
    ):
        path = plotter(rows, x, graph_dir)

        if path is not None:
            paths.append(path)

    return paths


def plot_latest_one_step_actor_critic_run(
    root_dir: str = RUNS_DIR,
) -> list[str]:
    run_root = Path(root_dir)
    run_glob = f"{ONE_STEP_ACTOR_CRITIC_RUN_NAME}_*"
    candidates = sorted(
        [
            path
            for path in run_root.glob(run_glob)
            if path.is_dir() and (path / "episodes.csv").exists()
        ],
        key=lambda path: path.stat().st_mtime,
    )

    if not candidates:
        raise FileNotFoundError(
            f"No {ONE_STEP_ACTOR_CRITIC_RUN_NAME} run with episodes.csv found in {root_dir}"
        )

    return plot_one_step_actor_critic_run(str(candidates[-1]))
