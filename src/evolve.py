import neat

import gzip
import pickle
import os

from helper.evaluate import neat_eval_genomes, neat_parallel_eval_genomes
import helper.visualize as visualize
from configs import evolve_config


def evolve(config_file, evolve_config):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    # p.add_reporter(neat.Checkpointer(5))

    #
    winner = p.run(
        neat_parallel_eval_genomes if evolve_config['parallel'] else neat_eval_genomes, evolve_config['n_generations'])

    # Pickle winner genome
    with gzip.open('evolve_results/winer', 'w', compresslevel=5) as f:
        data = (winner, config)
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    visualize.plot_stats(stats, ylog=False, view=True,
                         filename='evolve_results/avg_fitness.svg')


def run_evolve():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    evolve(config_path, evolve_config)


if __name__ == '__main__':
    run_evolve()
