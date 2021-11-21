# Forest-fire simulation with "AI" agent

![](animation/neat_simulation.gif)

:fire: - orange

:evergreen_tree: - green

:fire_engine: - blue

:black_circle: - brown

[Forest-fire model](https://en.wikipedia.org/wiki/Forest-fire_model) with [agent](###Agent), wich was evolved by [NEAT](https://neat-python.readthedocs.io/en/latest/). It is evolving neural network by genetic algorithm, where neural network maps environment state and agent position to action space.

![](evolve_results/avg_fitness.svg)

### Agent:
Has five actions:
* stay
* up
* down
* left
* right

If the agent at end of the action stays on the fire, extinguishes the tree.

### Results:
Number of extinguished trees in this environment setting (env. size: (16, 16), epochs: 50, , p: 0.5, f: 0.08):
* random agent - 8.6
* evolved NEAT agent - 24.58

In this setting, it looks like the NEAT agent waits for a nearby fire and then move.

## TODO
* clean code
* fix animation
* rewrite enviroment update
* play with parameters
* experiment with other eviroment rules
* change mlp for cnn