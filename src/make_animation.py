import imageio

import os

from helper.evaluate import eval_function, eval_baseline
from helper.utils import load_winner, clear_folder


def make_animation(image_folder, video_name):
    with imageio.get_writer(video_name, mode='I') as writer:
        for filename in os.listdir(image_folder):
            file_path = os.path.join(image_folder, filename)
            image = imageio.imread(file_path)
            writer.append_data(image)


def run_make_animation():
    print('\n ****** Neat animation ****** ')
    clear_folder('animation/images/neat')
    _ = eval_function(genome_config=load_winner(), save_env=True)
    make_animation('animation/images/neat',
                   'animation/neat_simulation.gif')

    print('\n ****** Baseline animation ****** ')
    clear_folder('animation/images/baseline')
    _ = eval_baseline(save_env=True)
    make_animation('animation/images/baseline',
                   'animation/baseline_simulation.gif')


if __name__ == '__main__':
    run_make_animation()
