import neat

import multiprocessing
from multiprocessing import Pool
from typing import Tuple

from env.forest_fire_enviroment import ForestFireEnviroment
from agents.firefighter import Firefigheter
from agents.baseline_policy import BaselinePolicy
from helper.utils import save_env_as_img


def eval_baseline(save_env: bool = False):
    from configs import env_config
    policy = BaselinePolicy()
    forest = ForestFireEnviroment(
        env_config['forest_size'], env_config['p_empty_tree'], env_config['p_tree_fire'])
    init_position = (env_config['forest_size'][0] //
                     2, env_config['forest_size'][1]//2)
    agent = Firefigheter(init_position, env_config['forest_size'], policy)

    for epoch in range(env_config['epochs']):
        forest.update_enviroment()
        forest.forest_state = agent.make_action(forest.forest_state)
        if save_env:

            save_env_as_img(agent.env_agent(forest.forest_state),
                            f'animation/images/baseline/img{epoch}.png')

    return agent.saved_trees


def eval_function(genome_config: Tuple, save_env: bool = False):
    from configs import env_config
    genome, config = genome_config
    genome.fitness = 0
    policy = neat.nn.FeedForwardNetwork.create(genome, config)
    forest = ForestFireEnviroment(
        env_config['forest_size'], env_config['p_empty_tree'], env_config['p_tree_fire'])
    init_position = (env_config['forest_size'][0] //
                     2, env_config['forest_size'][1]//2)
    agent = Firefigheter(init_position, env_config['forest_size'], policy)

    for epoch in range(env_config['epochs']):
        forest.update_enviroment()
        forest.forest_state = agent.make_action(forest.forest_state)
        if save_env:
            save_env_as_img(agent.env_agent(forest.forest_state),
                            f'animation/images/neat/img{epoch}.png')

    return agent.saved_trees


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness += eval_function((genome, config))


def parallel_eval_genomes(genomes, config):
    parallel_eval = ParallelEvaluator(
        multiprocessing.cpu_count(), eval_function)
    parallel_eval.evaluate(genomes, config)


class ParallelEvaluator(object):
    """ Fixed code from https://neat-python.readthedocs.io/en/latest/_modules/parallel.html#ParallelEvaluator"""

    def __init__(self, num_workers, eval_function, timeout=None):
        """
        eval_function should take one argument, a tuple of
        (genome object, config object), and return
        a single float (the genome's fitness).
        """
        self.num_workers = num_workers
        self.eval_function = eval_function
        self.timeout = timeout
        self.pool = Pool(num_workers)

    def __del__(self):
        self.pool.close()  # should this be terminate?
        self.pool.join()

    def evaluate(self, genomes, config):
        jobs = []
        for ignored_genome_id, genome in genomes:
            jobs.append(self.pool.apply_async(
                self.eval_function, [(genome, config)]))

        # assign the fitness back to each genome
        for job, (ignored_genome_id, genome) in zip(jobs, genomes):
            genome.fitness = job.get(timeout=self.timeout)
