from src.constants import BASE_DIR, CONF_PATH, DOSBOX_PATH, PLAYER_TEMPLATE_PATH, DONKEY_TEMPLATE_PATH
from src.game import run_training
 
 
def main():
    print("BASE_DIR:", BASE_DIR)
    print("CONF_PATH:", CONF_PATH)
    print("DOSBOX_PATH:", DOSBOX_PATH)
    print("PLAYER_TEMPLATE_PATH:", PLAYER_TEMPLATE_PATH)
    print("DONKEY_TEMPLATE_PATH:", DONKEY_TEMPLATE_PATH)
 
    run_training(num_episodes=20000, step_interval=0.2)
 
if __name__ == "__main__":
    main()