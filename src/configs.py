env_config = {
    # Also change num_inputs in config-feedforward.txt to height*width+2
    'forest_size': (16, 16),
    # Also change fitness_threshold in config-feedforward.txt
    'epochs': 10,
    'p_empty_tree': 0.5,
    'p_tree_fire': 0.08,
}

evolve_config = {
    'n_generations': 20,
    'parallel': True,
}

monte_carlo_config = {'max_simulation': 50,
                      'treshold': 2,
                      'verbose': False,
                      'warmup': 5,
                      }
