import random
import math
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch

LOG_IMPOSSIBLE_VALUE = -1000
LOG_NOISE_VALUE = -10
LOG_NOISE_VALUE_DELETED = -10


def logsumexp(x, y):
    return scipy.special.logsumexp(x, y)


def exp_normalize(x):
    x = np.array(x)
    b = x.max()
    y = np.exp(x - b)
    return y / y.sum()


def torch_exp_normalize(x):
    b = x.max()
    y = torch.exp(x - b)
    return (y / y.sum()).numpy()


def random_color():
    return [random.random() for _ in range(3)]


def draw_squares(squares, ax):
    # Add squares to the plot
    for square in squares:
        color = random_color()  # Generate a random color for each square
        rect = patches.Rectangle((square['x'], square['y']),
                                 square['w'],
                                 square['h'],
                                 linewidth=1,
                                 edgecolor=color,
                                 facecolor=color)
        ax.add_patch(rect)
        ax.text(square['x'] + square['w'] / 2,
                square['y'] + square['h'] / 2,
                square['name'],
                color='white',
                ha='center',
                va='center',
                fontsize=12)

    # Set the limits of the plot
    ax.set_xlim(0, max([square['x'] + square['w'] for square in squares]) + 1)
    ax.set_ylim(0, max([square['y'] + square['h'] for square in squares]) + 1)

    # Set aspect ratio to be equal
    ax.set_aspect('equal')

    ax.invert_yaxis()


def display_obj_lists(obj_lists,
                      indices=None,
                      title=None,
                      nrows=None,
                      ncols=None,
                      grid_size=16):
    if indices is None:
        indices = range(len(obj_lists))

    if nrows is None or ncols is None:
        ncols = 2
        nrows = int(math.ceil(len(indices) / ncols))
    fig, axes = plt.subplots(nrows, ncols)
    if nrows == 1:
        axes = np.asarray([axes])

    offset = np.asarray([1, 1])
    for ct, idx in enumerate(indices):
        obj_list = obj_lists[idx]

        ax = axes[ct // ncols, ct % ncols]

        # create squares
        draw_squares([{
            'name': obj.obj_type[:1].upper(),
            'x': obj.x / 10,
            'y': obj.y / 10,
            'w': obj.w / 10,
            'h': obj.h / 10
        } for obj in obj_list], ax)

        # draw gridlines
        ax.grid(which='major',
                axis='both',
                linestyle='-',
                color='white',
                linewidth=1)
        ax.set_title(f'Index: {idx}', fontsize=8)

        # turn off ticks
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)

    for ct in range(len(indices), nrows * ncols):
        ax = axes[ct // ncols, ct % ncols]
        ax.axis('off')

    if title is not None:
        plt.subtitle(title, fontsize=10)
    plt.show()


def nice_display_obj_lists(obj_lists,
                           indices=None,
                           title=None,
                           nrows=None,
                           ncols=None,
                           grid_size=16,
                           game_name=None):
    raise NotImplementedError
