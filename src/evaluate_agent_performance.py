import numpy as np
import scipy.stats

import json
import timeit

from agents.policy import NEATPolicy, LazyPolicy, RadnomPolicy
from helper.evaluate import eval_policy
from helper.utils import load_winner_genome
from configs import monte_carlo_config, env_config


def monte_carlo_estimation(eval_policy, eval_policy_params,
                           max_simulation: int, treshold: float,
                           verbose, warmup=5):
    def _confidence_interval(L, Q, n):
        mean = L/n
        variance = Q/n - (L/n)**2
        lenght = scipy.stats.norm.ppf(0.975)*variance/(n**(1/2))
        lower = mean - lenght
        upper = mean + lenght
        return lower, upper, lenght

    def _compute_results(lower, upper, length, n):
        return {'estimate': upper - length, 'confidence_interval': (lower, upper),
                'confidence_interval_lenght': length,  'number_of_simulation': n}

    length = np.inf
    L = 0
    Q = 0
    for n in range(1, max_simulation+1):
        if verbose:
            start = timeit.default_timer()

        etimate = eval_policy(**eval_policy_params)

        if verbose:
            stop = timeit.default_timer()

        L += etimate
        Q += etimate**2

        lower, upper, length = _confidence_interval(L, Q, n)
        if verbose:
            results = _compute_results(lower, upper, length, n)
            results['time'] = stop - start
            print(results)

        if length <= treshold and n >= warmup:
            return _compute_results(lower, upper, length, n)

    return _compute_results(lower, upper, length, n)


def run_eval_agents():
    genome, config = load_winner_genome()
    neat_eval_policy_params = {
        'policy_param': (NEATPolicy, {'genome': genome, 'config': config}),
        'env_config': env_config}
    lazy_eval_policy_params = {
        'policy_param': (LazyPolicy, {'n_actions': 5}),
        'env_config': env_config}
    random_policy_params = {
        'policy_param': (RadnomPolicy, {'n_actions': 5}),
        'env_config': env_config}

    name_param = [('NEATPolicy', neat_eval_policy_params),
                  ('LazyPolicy', lazy_eval_policy_params),
                  ('RadnomPolicy', random_policy_params)]
    for name_eval_policy_params in name_param:
        name, eval_policy_params = name_eval_policy_params
        print(f'\n ****** {name} ****** ')
        results = monte_carlo_estimation(
            eval_policy, eval_policy_params, **monte_carlo_config)
        with open(f'evolve_results/evaluation/{name}.json', 'w') as fp:
            json.dump(results, fp)


if __name__ == '__main__':
    run_eval_agents()
