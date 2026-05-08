import os
import time
import subprocess

import numpy as np
import pyautogui

from src.config import (
    DOSBOX_PATH,
    CONF_PATH,
    PLAYER_TEMPLATE_PATH,
    DONKEY_TEMPLATE_PATH,
    IMAGE_TEMPLATE_DIR,
    AgentMode,
    STATE_SIZE
)
from src.detection import load_score_templates
from src.env.episode import run_episode
from src.env.logging import format_episode_metrics
from src.window import find_dosbox_window, activate_window, get_capture_region
from src.utils.seed_init import set_seed

from agents.dgn_agent import DQNAgent
from agents.actor_critic.agents.episodic import EpisodicActorCriticAgent

def validate_paths():
    required = {
        "DOSBox.exe": DOSBOX_PATH,
        "dosbox.conf": CONF_PATH,
        "player_template.png": PLAYER_TEMPLATE_PATH,
        "donkey_template.png": DONKEY_TEMPLATE_PATH,
    }

    for name, path in required.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} not found: {path}")
        

def run_training(
    mode:AgentMode,
    num_episodes: int = 20000,
    step_interval: float = 0.15
):
    set_seed(122)
    validate_paths()
    
    score_templates_dir = os.path.join(
        IMAGE_TEMPLATE_DIR,
        "score_templates",
    )

    templates = load_score_templates(score_templates_dir)

    if not templates:
        raise RuntimeError(
            "No score templates found. "
            f"Put digit_0.png..digit_9.png into {score_templates_dir}"
        )

    process = None
    agent = None

    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    try:
        print("Starting DOSBox...")
        process = subprocess.Popen([DOSBOX_PATH, "-conf", CONF_PATH])

        time.sleep(3)

        window = find_dosbox_window(timeout=10)
        activate_window(window)

        print(
            f"Window: left={window.left} top={window.top} "
            f"w={window.width} h={window.height}"
        )

        region = get_capture_region(window)


        if mode == AgentMode.DQN:
            agent = DQNAgent(flag_double=False)
        elif mode == AgentMode.DOUBLE_DQN:
            agent = DQNAgent(flag_double=True)
        else:
            agent = EpisodicActorCriticAgent(
                state_size=STATE_SIZE,
                hidden_layers=[64, 64],
                gamma=0.97,
                lr=0.0003,
                entropy_coef=0.001,
                normalize_returns=True,
            )

        pyautogui.press("space")
        time.sleep(1)

        episode_rewards: list[float] = []

        for ep in range(num_episodes):
            total_reward, _ = run_episode(
                region=region,
                templates=templates,
                agent=agent,
                mode=mode,
                episode_idx=ep,
                step_interval=step_interval,
            )

            if mode == AgentMode.ACTOR_CRITIC:
                agent.finish_episode()

            if hasattr(agent, "save") and (ep + 1) % 50 == 0:
                agent.save(os.path.join(checkpoint_dir, f"agent1_ep_{ep + 1}.pt"))
            episode_rewards.append(total_reward)

            avg_last_10 = float(np.mean(episode_rewards[-10:]))


            print(
                f"[TRAIN] episode={ep} "
                f"reward={total_reward:.1f} "
                f"avg_last_10={avg_last_10:.1f}"
                f"{format_episode_metrics(mode, agent)}"
            )

    except KeyboardInterrupt:
        print("Training stopped by user.")

    finally:
        if agent is not None and hasattr(agent, "save"):
            agent.save(os.path.join(checkpoint_dir, "agent1_last.pt"))

        if process is not None:
            process.terminate()