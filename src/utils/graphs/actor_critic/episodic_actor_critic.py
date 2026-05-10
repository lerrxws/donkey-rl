import os

from src.utils.graphs.actor_critic.actor_critic import (
    load_actor_critic_episodes,
    plot_episode_steps,
    plot_entropy,
    plot_total_reward,
    plot_value,
)
from src.utils.graphs.base import (
    BaseRunPlotter,
    has_columns,
    numeric_column,
    save_line_plot,
)


def _plot_episode_metric(
    rows: list[dict[str, str]],
    x,
    graph_dir: str,
    column: str,
    title: str,
    ylabel: str,
    output_name: str,
) -> str | None:
    if not has_columns(rows, ["episode", column]):
        return None

    return save_line_plot(
        x=x,
        series={column: numeric_column(rows, column)},
        title=title,
        ylabel=ylabel,
        path=os.path.join(graph_dir, output_name),
    )


def _plot_value_vs_target(
    rows: list[dict[str, str]],
    x,
    graph_dir: str,
) -> str | None:
    if has_columns(rows, ["episode", "value", "target"]):
        value_column = "value"
        target_column = "target"
    elif has_columns(rows, ["episode", "mean_value", "mean_target"]):
        value_column = "mean_value"
        target_column = "mean_target"
    else:
        return None

    return save_line_plot(
        x=x,
        series={
            value_column: numeric_column(rows, value_column),
            "target_G": numeric_column(rows, target_column),
        },
        title="Value vs target G",
        ylabel="value / G",
        path=os.path.join(graph_dir, "value_vs_target_g.png"),
    )


class EpisodicActorCriticRunPlotter(BaseRunPlotter):
    def plot(self, run_dir: str) -> list[str]:
        rows, x, graph_dir = load_actor_critic_episodes(run_dir)
        paths = []

        for plotter in (
            plot_total_reward,
            plot_episode_steps,
            lambda r, episodes, output_dir: _plot_episode_metric(
                rows=r,
                x=episodes,
                graph_dir=output_dir,
                column="actor_loss",
                title="Actor loss per episode",
                ylabel="actor_loss",
                output_name="actor_loss.png",
            ),
            lambda r, episodes, output_dir: _plot_episode_metric(
                rows=r,
                x=episodes,
                graph_dir=output_dir,
                column="critic_loss",
                title="Critic loss per episode",
                ylabel="critic_loss",
                output_name="critic_loss.png",
            ),
            _plot_value_vs_target,
            lambda r, episodes, output_dir: _plot_episode_metric(
                rows=r,
                x=episodes,
                graph_dir=output_dir,
                column="advantage",
                title="Mean advantage per episode",
                ylabel="advantage",
                output_name="advantage.png",
            ),
            plot_entropy,
            plot_value,
        ):
            path = plotter(rows, x, graph_dir)

            if path is not None:
                paths.append(path)

        return paths


def plot_episodic_actor_critic_run(run_dir: str) -> list[str]:
    return EpisodicActorCriticRunPlotter().plot(run_dir)
