import json
import os
import time
from abc import ABC
from collections import defaultdict
from typing import Any

import numpy as np

from src.constants import RUNS_DIR
from src.utils.csv_logger import CSVLogger


class BaseTrainingTracker(ABC):
    STEP_FIELDS: list[str] = []
    EPISODE_FIELDS: list[str] = []
    EPISODE_SUMMARIES: dict[str, tuple[str, ...]] = {}

    def __init__(
        self,
        run_name: str,
        config: dict[str, Any],
        root_dir: str = RUNS_DIR,
        save_steps: bool = True,
    ):
        timestamp = time.strftime("%Y_%m_%d_%H_%M_%S")

        self.run_name = run_name
        self.config = config
        self.root_dir = root_dir
        self.save_steps = save_steps

        self.run_dir = os.path.join(root_dir, f"{run_name}_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)

        self.config_path = os.path.join(self.run_dir, "config.json")
        self.steps_path = os.path.join(self.run_dir, "steps.csv")
        self.episodes_path = os.path.join(self.run_dir, "episodes.csv")

        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, default=str)

        self.step_fields = ["episode", "step"] + self.STEP_FIELDS
        self.episode_fields = (
            ["episode"]
            + self.EPISODE_FIELDS
            + self.__build_episode_metric_fields()
        )

        self.step_logger = (
            CSVLogger(self.steps_path, self.step_fields)
            if save_steps
            else None
        )

        self.episode_logger = CSVLogger(
            self.episodes_path,
            self.episode_fields,
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
            **self.__compute_metric_summary(),
        }

        self.episode_logger.write(row)
        self.episode_buffers.clear()

    def close(self) -> None:
        if self.step_logger is not None:
            self.step_logger.close()

        self.episode_logger.close()

    def __update_episode_buffers(self, row: dict[str, Any]) -> None:
        for key in self.EPISODE_SUMMARIES:
            value = row.get(key)

            if isinstance(value, bool):
                continue

            if isinstance(value, (int, float)):
                self.episode_buffers[key].append(float(value))

    def __compute_metric_summary(self) -> dict[str, float]:
        summary = {}

        for key, summary_types in self.EPISODE_SUMMARIES.items():
            values = self.episode_buffers.get(key, [])

            if not values:
                continue

            arr = np.asarray(values, dtype=np.float32)

            if "mean" in summary_types:
                summary[f"{key}_mean"] = float(arr.mean())

            if "std" in summary_types:
                summary[f"{key}_std"] = float(arr.std())

            if "abs_mean" in summary_types:
                summary[f"{key}_abs_mean"] = float(np.abs(arr).mean())

            if "min" in summary_types:
                summary[f"{key}_min"] = float(arr.min())

            if "max" in summary_types:
                summary[f"{key}_max"] = float(arr.max())

        return summary

    def __build_episode_metric_fields(self) -> list[str]:
        fields = []

        for key, summary_types in self.EPISODE_SUMMARIES.items():
            for summary_type in summary_types:
                fields.append(f"{key}_{summary_type}")

        return fields

    def __mean_last(
        self,
        values: list[float] | list[int],
        n: int,
    ) -> float:
        if not values:
            return 0.0

        return float(np.mean(values[-n:]))
