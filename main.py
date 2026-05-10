from src.training import run_training
 
if __name__ == "__main__":
    run_training(
        mode="double_dqn",
        num_episodes=20000, 
        step_interval=0.2
    )