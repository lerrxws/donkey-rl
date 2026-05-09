from src.utils.graphs.actor_critic.actor_critic import (
    load_actor_critic_episodes,
    plot_episode_steps,
    plot_entropy,
    plot_total_reward,
    plot_value,
)
from src.utils.graphs.base import BaseRunPlotter


class EpisodicActorCriticRunPlotter(BaseRunPlotter):
    def plot(self, run_dir: str) -> list[str]:
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


def plot_episodic_actor_critic_run(run_dir: str) -> list[str]:
    return EpisodicActorCriticRunPlotter().plot(run_dir)
