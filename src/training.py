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
    STATE_SIZE,
    HIDDEN_LAYERS_SIZE,
    ACTION_SIZE,
    ONE_STEP_ACTOR_CRITIC_RUN_NAME,
    Q_LEARNING_RUN_NAME,
    DOUBLE_Q_LEARNING_RUN_NAME,
    RUNS_DIR
)
log_path = "training_logs.csv"
from src.detection import load_score_templates
from src.env.episode import run_episode
from src.env.logging import format_episode_metrics
from src.window import find_dosbox_window, activate_window, get_capture_region
from src.utils.seed_init import set_seed

from src.utils.metrics.actor_critic.one_step_actor_critic import OneStepActorCriticTracker
from src.utils.metrics.q_learning.q_learning import DQNTrainingTracker
from src.utils.graphs import plot_one_step_actor_critic_run

from src.agents.q_learning.perform_action import perform_action
from src.agents.actor_critic.agents.one_step import OneStepActorCriticAgent
from src.agents.q_learning.dgn_agent import DQNAgent

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
    start_time=time.perf_counter()
    set_seed(125)
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

        region = get_capture_region(window)


        if mode in (AgentMode.DQN,AgentMode.DOUBLE_DQN):
            if mode == AgentMode.DQN:
                agent = DQNAgent(
                    state_size=STATE_SIZE,
                    action_size=ACTION_SIZE,
                    hidden_layers=HIDDEN_LAYERS_SIZE,
                    flag_double=False
                )
                run_name=Q_LEARNING_RUN_NAME

            else:
                agent = DQNAgent(
                    state_size=STATE_SIZE,
                    action_size=ACTION_SIZE,
                    hidden_layers=HIDDEN_LAYERS_SIZE,
                    flag_double=True
                )
                run_name=DOUBLE_Q_LEARNING_RUN_NAME

            tracker = DQNTrainingTracker(
                run_name=run_name,
                root_dir=RUNS_DIR,
                config={
                    "algorithm": run_name,
                    "state_size": STATE_SIZE,
                    "action_size": ACTION_SIZE,
                    "hidden_layer": HIDDEN_LAYERS_SIZE,
                    "epsilon": 1.0,
                    "epsilon_min": 0.05,
                    "epsilon_decay": 0.9995,
                    "gamma": 0.99,
                    "target_update_every": 500,
                    "step_interval": step_interval
                },
                save_steps=True,
            )
        else:
            agent = OneStepActorCriticAgent(
                state_size=STATE_SIZE,
                action_size=ACTION_SIZE,
                hidden_layers=HIDDEN_LAYERS_SIZE,
                gamma=0.97,
                actor_lr=0.0003,
                critic_lr=0.0003,
                entropy_coef=0.01,
                reward_scale=100.0,
                max_grad_norm=1.0,
            )

            tracker = OneStepActorCriticTracker(
                run_name=ONE_STEP_ACTOR_CRITIC_RUN_NAME,
                root_dir=RUNS_DIR,
                config={
                    "algorithm": ONE_STEP_ACTOR_CRITIC_RUN_NAME,
                    "state_size": STATE_SIZE,
                    "action_size": ACTION_SIZE,
                    "hidden_layer":HIDDEN_LAYERS_SIZE,
                    "gamma": 0.97,
                    "actor_lr": 0.0003,
                    "critic_lr": 0.0003,
                    "entropy_coef": 0.02,
                    "reward_scale": 100.0,
                    "max_grad_norm": 1.0,
                    "step_interval": step_interval
                },
                save_steps=True,
            )


        pyautogui.press("space")
        time.sleep(1)

        episode_rewards: list[float] = []
        episode_logs = []
        for ep in range(num_episodes):
            episode_info = run_episode(
                region=region,
                templates=templates,
                agent=agent,
                mode=mode,
                episode_idx=ep,
                step_interval=step_interval,
                tracker=tracker
            )

            if mode == AgentMode.ACTOR_CRITIC:
                agent.finish_episode()
            
            if hasattr(agent, "save") and (ep + 1) % 50 == 0:
                agent.save(os.path.join(checkpoint_dir, f"agent1_ep_{ep + 1}.pt"))
            episode_rewards.append(episode_info.get("total_reward"))

            avg_last_10 = float(np.mean(episode_rewards[-10:]))

            episode_info["avg_reward_last_10"] = avg_last_10
            episode_logs.append(episode_info)
            
            tracker.record_episode(ep,episode_info)
            print(
                f"[TRAIN] episode={ep} "
                f"reward={episode_info.get('total_reward'):.1f} "
                f"avg_last_10={avg_last_10:.1f}"
                f"{format_episode_metrics(mode, agent)}"
            )

    except KeyboardInterrupt:
        print("Training stopped by user.")

    finally:
        end_time=time.perf_counter()
        elapsed=end_time-start_time
        print(
            f"Training was running for {elapsed:.1f} seconds "
            f"({elapsed / 60:.2f} minutes)"
        )
        tracker.close()
        if agent is not None and hasattr(agent, "save"):
            agent.save(os.path.join(checkpoint_dir, "agent1_last.pt"))
        if process is not None:
            process.terminate()