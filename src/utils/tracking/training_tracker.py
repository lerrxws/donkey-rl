import json
import os
import time
from collections import defaultdict
from typing import Any

import numpy as np

from src.utils.tracking.csv_logger import CSVLogger


STEP_FIELDS = [
    "episode",  # Episode index.
    "step",  # Step index inside the episode.
    "reward",  # Raw reward received at this step.
    "total_reward",  # Cumulative reward collected in the episode so far.
    "action",  # Action selected by the agent.
    "done",  # Whether the episode ended after this step.

    "actor_loss",  # Actor loss for the current update.
    "critic_loss",  # Critic loss for the current update.
    "td_error",  # TD error used as the learning signal.
    "entropy",  # Policy entropy showing how random the action choice is.
    "value",  # Critic prediction V(s) for the current state.
    "target",  # Training target for the Critic.
    "scaled_reward",  # Reward after scaling before training.

    "prob_no_jump",  # Policy probability of choosing no jump.
    "prob_jump",  # Policy probability of choosing jump.
    "selected_action_prob",  # Probability of the action actually selected.

    "danger",  # Whether the donkey is in a dangerous position.
    "same_line",  # Whether the player and donkey are in the same lane.
    "good_jump",  # Whether the jump was useful.
    "bad_jump",  # Whether the jump was unnecessary.
    "missed_jump",  # Whether the agent failed to jump in danger.
    "side_jump",  # Whether the agent jumped while donkey was in another lane.
    "crash",  # Whether a crash happened.
]


EPISODE_FIELDS = [
    "episode",  # Episode index.
    "total_reward",  # Final total reward for the episode.
    "avg_reward_10",  # Mean reward over the last 10 episodes.
    "avg_reward_50",  # Mean reward over the last 50 episodes.
    "episode_steps",  # Number of steps survived in the episode.

    "actor_loss_mean",  # Mean Actor loss over the episode.
    "actor_loss_std",  # Variability of Actor loss over the episode.
    "critic_loss_mean",  # Mean Critic loss over the episode.
    "critic_loss_std",  # Variability of Critic loss over the episode.

    "td_error_mean",  # Mean signed TD error over the episode.
    "td_error_abs_mean",  # Mean absolute TD error showing Critic error size.
    "td_error_std",  # Variability of TD error over the episode.
    "td_error_min",  # Smallest TD error in the episode.
    "td_error_max",  # Largest TD error in the episode.

    "entropy_mean",  # Mean policy entropy over the episode.
    "entropy_std",  # Variability of policy entropy over the episode.

    "value_mean",  # Mean Critic value prediction V(s).
    "target_mean",  # Mean Critic training target.
    "scaled_reward_mean",  # Mean scaled reward used for training.

    "prob_no_jump_mean",  # Mean policy probability of no jump.
    "prob_jump_mean",  # Mean policy probability of jump.
    "selected_action_prob_mean",  # Mean probability of selected actions.

    "action_0_count",  # Number of no-jump actions.
    "action_1_count",  # Number of jump actions.
    "action_0_rate",  # Fraction of no-jump actions.
    "action_1_rate",  # Fraction of jump actions.

    "good_jumps",  # Number of useful jumps.
    "bad_jumps",  # Number of unnecessary jumps.
    "missed_jumps",  # Number of missed dangerous situations.
    "side_jumps",  # Number of jumps when donkey was in another lane.

    "crash",  # Whether the episode ended with a crash.
    "crash_rate_50",  # Crash rate over the last 50 episodes.
]

class TrainingTracker:
    def __init__(
        self,
        run_name: str,
        config: dict[str, Any],
        root_dir: str = "data/runs",
        save_steps: bool = True,
    ):
        timestamp = time.strftime("%Y_%m_%d_%H_%M_%S")

        self.run_name = run_name
        self.run_dir = os.path.join(root_dir, f"{run_name}_{timestamp}")
        self.save_steps = save_steps

        os.makedirs(self.run_dir, exist_ok=True)

        self.config_path = os.path.join(self.run_dir, "config.json")
        self.steps_path = os.path.join(self.run_dir, "steps.csv")
        self.episodes_path = os.path.join(self.run_dir, "episodes.csv")

        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)

        self.step_logger = (
            CSVLogger(self.steps_path, STEP_FIELDS)
            if save_steps
            else None
        )

        self.episode_logger = CSVLogger(
            self.episodes_path,
            EPISODE_FIELDS,
        )

        self.episode_buffers: dict[str, list[float]] = defaultdict(list)
        self.reward_history: list[float] = []
        self.crash_history: list[int] = []

    def record_step(
        self,
        episode: int,
        step: int,
        data: dict[str, Any],
    ) -> None:
        row = {
            "episode": episode,
            "step": step,
            **data,
        }

        self.__update_episode_buffers(row)

        if self.step_logger is not None:
            self.step_logger.write(row)

    def record_episode(
        self,
        episode: int,
        data: dict[str, Any],
    ) -> None:
        total_reward = float(data.get("total_reward", 0.0))
        crash = int(data.get("crash", 0))

        self.reward_history.append(total_reward)
        self.crash_history.append(crash)

        row = {
            "episode": episode,
            **data,
            "avg_reward_10": self.__mean_last(self.reward_history, 10),
            "avg_reward_50": self.__mean_last(self.reward_history, 50),
            "crash_rate_50": self.__mean_last(self.crash_history, 50),
            **self.__compute_episode_summary(),
        }

        self.episode_logger.write(row)
        self.episode_buffers.clear()

    def close(self) -> None:
        if self.step_logger is not None:
            self.step_logger.close()

        self.episode_logger.close()

    def __update_episode_buffers(self, row: dict[str, Any]) -> None:
        numeric_keys = [
            "actor_loss",
            "critic_loss",
            "td_error",
            "entropy",
            "value",
            "target",
            "scaled_reward",
            "prob_no_jump",
            "prob_jump",
            "selected_action_prob",
        ]

        for key in numeric_keys:
            value = row.get(key)

            if isinstance(value, bool):
                continue

            if isinstance(value, (int, float)):
                self.episode_buffers[key].append(float(value))

    def __compute_episode_summary(self) -> dict[str, float]:
        summary = {}

        self.__add_mean_std(summary, "actor_loss")
        self.__add_mean_std(summary, "critic_loss")
        self.__add_mean_std(summary, "entropy")

        self.__add_mean(summary, "value")
        self.__add_mean(summary, "target")
        self.__add_mean(summary, "scaled_reward")
        self.__add_mean(summary, "prob_no_jump")
        self.__add_mean(summary, "prob_jump")
        self.__add_mean(summary, "selected_action_prob")

        td_errors = self.episode_buffers.get("td_error", [])

        if td_errors:
            arr = np.asarray(td_errors, dtype=np.float32)

            summary["td_error_mean"] = float(arr.mean())
            summary["td_error_abs_mean"] = float(np.abs(arr).mean())
            summary["td_error_std"] = float(arr.std())
            summary["td_error_min"] = float(arr.min())
            summary["td_error_max"] = float(arr.max())

        return summary

    def __add_mean_std(
        self,
        summary: dict[str, float],
        key: str,
    ) -> None:
        values = self.episode_buffers.get(key, [])

        if not values:
            return

        arr = np.asarray(values, dtype=np.float32)

        summary[f"{key}_mean"] = float(arr.mean())
        summary[f"{key}_std"] = float(arr.std())

    def __add_mean(
        self,
        summary: dict[str, float],
        key: str,
    ) -> None:
        values = self.episode_buffers.get(key, [])

        if not values:
            return

        arr = np.asarray(values, dtype=np.float32)

        summary[f"{key}_mean"] = float(arr.mean())

    def __mean_last(
        self,
        values: list[float] | list[int],
        n: int,
    ) -> float:
        if not values:
            return 0.0

        return float(np.mean(values[-n:]))