from src.config import AgentMode
def format_episode_metrics(mode: AgentMode, agent) -> str:
    if mode in (AgentMode.DQN, AgentMode.DOUBLE_DQN):
        epsilon = getattr(agent, "epsilon", None)
        if epsilon is None:
            return ""
        return f" | epsilon={epsilon:.3f}"

    m = getattr(agent, "last_metrics", None)
    if m is None:
        return ""

    parts = []

    if "actor_loss" in m:
        parts.append(f"actor={m['actor_loss']:+.4f}")

    if "critic_loss" in m:
        parts.append(f"critic={m['critic_loss']:+.4f}")

    if "td_error" in m:
        parts.append(f"td={m['td_error']:+.4f}")

    if "value" in m:
        parts.append(f"V={m['value']:+.4f}")

    if "target" in m:
        parts.append(f"target={m['target']:+.4f}")

    if "entropy_mean" in m:
        parts.append(f"H={m['entropy_mean']:.4f}")

    if "scaled_reward" in m:
        parts.append(f"r_scaled={m['scaled_reward']:+.4f}")

    if "prob_no_jump" in m and "prob_jump" in m:
        parts.append(
            f"probs=[no_jump:{m['prob_no_jump']:.3f},jump:{m['prob_jump']:.3f}]"
        )

    return " | " + " | ".join(parts) if parts else ""
