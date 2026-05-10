from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class MetricRecord:
    def to_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in asdict(self).items()
            if value is not None
        }


@dataclass(frozen=True)
class DQNStepRecord(MetricRecord):
    loss: float | None = None
    epsilon: float | None = None


@dataclass(frozen=True)
class OneStepActorCriticStepRecord(MetricRecord):
    actor_loss: float | None = None
    critic_loss: float | None = None
    td_error: float | None = None
    advantage: float | None = None
    entropy_mean: float | None = None
    value: float | None = None
    next_value: float | None = None
    target: float | None = None
    scaled_reward: float | None = None
    prob_no_jump: float | None = None
    prob_jump: float | None = None
    selected_action_prob: float | None = None
