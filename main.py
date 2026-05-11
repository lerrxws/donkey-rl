from src.training import run_training
 
if __name__ == "__main__":
    seeds=[64,164,264,421]
    hidden_layer_sizes=[[64],[64,64,64]]
    modes=["dqn","double_dqn"]
    for mode in modes:
        for seed in seeds:
            run_training(
                mode=mode,
                num_episodes=20000, 
                step_interval=0.2,
                number_of_seed=seed,
                hidden_layers_size=[64,64]
            )
    
    for mode in modes:
        for hidden_layer_size in hidden_layer_sizes:
            run_training(
                mode=mode,
                num_episodes=20000, 
                step_interval=0.2,
                number_of_seed=164,
                hidden_layers_size=hidden_layer_size
            )