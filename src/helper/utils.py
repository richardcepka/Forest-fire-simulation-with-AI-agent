import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import random
import gzip
import pickle
import os
import shutil


def set_seed(seed=10):
    np.random.seed(seed)
    random.seed(seed)


def clear_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def load_winner_genome():
    with gzip.open('evolve_results/winer') as f:
        genome, config = pickle.load(f)
    return genome, config


def save_env_as_img(env, filename):
    img = env[0].numpy()
    colors = [(0.2, 0, 0), (0, 0.5, 0), 'orange', 'blue']
    cmap = matplotlib.colors.ListedColormap(colors, name='colors', N=None)
    plt.imshow(img, cmap=cmap)
    plt.axis('off')
    plt.savefig(filename)
    plt.close()
