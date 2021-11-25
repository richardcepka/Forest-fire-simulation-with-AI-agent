import imageio

import os

from agents.policy import NEATPolicy, LazyPolicy, RadnomPolicy
from helper.evaluate import eval_policy
from helper.utils import load_winner_genome, clear_folder
from configs import env_config


def make_animation(image_folder, video_name):
    with imageio.get_writer(video_name, mode='I') as writer:
        for filename in os.listdir(image_folder):
            file_path = os.path.join(image_folder, filename)
            image = imageio.imread(file_path)
            writer.append_data(image)


def run_make_animation():
    genome, config = load_winner_genome()
    neat_eval_policy_params = {
        'policy_param': (NEATPolicy, {'genome': genome, 'config': config}),
        'env_config': env_config,
        'save_env': True}
    lazy_eval_policy_params = {
        'policy_param': (LazyPolicy, {'n_actions': 5}),
        'env_config': env_config,
        'save_env': True}
    random_policy_params = {
        'policy_param': (RadnomPolicy, {'n_actions': 5}),
        'env_config': env_config,
        'save_env': True}

    name_param = [('NEATPolicy', neat_eval_policy_params),
                  #('LazyPolicy', lazy_eval_policy_params),
                  #('RadnomPolicy', random_policy_params)
                  ]
    for name_eval_policy_params in name_param:
        name, eval_policy_params = name_eval_policy_params
        print(f'\n ****** {name} animation ****** ')
        clear_folder(f'animation/images/{name}')
        _ = eval_policy(**eval_policy_params)
        make_animation(f'animation/images/{name}',
                       f'animation/{name}_simulation.gif')


if __name__ == '__main__':
    run_make_animation()
