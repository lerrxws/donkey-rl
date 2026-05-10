from src.config import AgentMode
from src.training import run_training
 
if __name__ == "__main__":
    run_training(
        mode=AgentMode.EPISODIC_ACTOR_CRITIC,
        num_episodes=20000, 
        step_interval=0.2
    )
