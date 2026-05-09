from src.config import AgentMode
from src.utils.metrics import OneStepActorCriticTracker

def track_step(
    tracker: OneStepActorCriticTracker | None,
    mode: AgentMode,
    agent,
    episode_idx: int,
    step: int,
    reward: float,
    total_reward: float,
    action: int,
    done: bool,
    flags: dict,
    good_jump: bool,
    bad_jump: bool,
    missed_jump: bool,
    side_jump: bool,
    crash_detected: bool,
) -> None:
    if tracker is None or mode != AgentMode.ACTOR_CRITIC:
        return

    metrics = getattr(agent, "last_metrics", None) or {}

    step_data = {
        "reward": float(reward),
        "total_reward": float(total_reward),
        "action": int(action),
        "done": int(done),

        "actor_loss": metrics.get("actor_loss"),
        "critic_loss": metrics.get("critic_loss"),
        "td_error": metrics.get("td_error"),
        "advantage": metrics.get("advantage"),
        "entropy_mean": metrics.get("entropy_mean"),
        "value": metrics.get("value"),
        "next_value": metrics.get("next_value"),
        "target": metrics.get("target"),
        "scaled_reward": metrics.get("scaled_reward"),
        "prob_no_jump": metrics.get("prob_no_jump"),
        "prob_jump": metrics.get("prob_jump"),
        "selected_action_prob": metrics.get("selected_action_prob"),

        "danger": int(flags["danger"]),
        "same_line": int(flags["same_line"]),
        "good_jump": int(good_jump),
        "bad_jump": int(bad_jump),
        "missed_jump": int(missed_jump),
        "side_jump": int(side_jump),
        "crash": int(crash_detected),
    }

    tracker.record_step(
        episode=episode_idx,
        step=step,
        data=step_data,
    )


def track_episode(
    tracker: OneStepActorCriticTracker | None,
    mode: AgentMode,
    episode_idx: int,
    total_reward: float,
    step: int,
    action_counts: dict[int, int],
    good_jump_count: int,
    bad_jump_count: int,
    missed_jump_count: int,
    side_jump_count: int,
    crash_detected: bool,
) -> None:
    if tracker is None or mode != AgentMode.ACTOR_CRITIC:
        return

    episode_steps = max(1, step)

    tracker.record_episode(
        episode=episode_idx,
        data={
            "total_reward": float(total_reward),
            "episode_steps": int(episode_steps),

            "action_0_count": int(action_counts[0]),
            "action_1_count": int(action_counts[1]),
            "action_0_rate": float(action_counts[0] / episode_steps),
            "action_1_rate": float(action_counts[1] / episode_steps),

            "good_jumps": int(good_jump_count),
            "bad_jumps": int(bad_jump_count),
            "missed_jumps": int(missed_jump_count),
            "side_jumps": int(side_jump_count),

            "crash": int(crash_detected),
        },
    )