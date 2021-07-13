"""Usage:    visualize_k_means.py --diversity_file=diversity_file --save_file=save_file
             visualize_k_means.py --help | -help | -h

Visualize k_means on a smaller subset using networkx 
with Haussdroff distance as the diatance metric
Arguments:
  diversity_file      a file that contains diversity values for the samples
  save_file           path for saving image

Options:
  -h --help -h
"""

import random
import numpy as np
import copy
from pathlib import Path
import os
from docopt import docopt
from PIL import Image
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial.distance import directed_hausdorff
from k_means import find_diverse_files

def create_graph(diversity_file, save_file):
    """
    Function to visualize and save k_means results as a graph using networkx
    Args:
        diversity_file: File that has the dim reduced vectors
        save_file: file name for saving the generated image
    """
    sample_size = 4
    pool_size = 12
    diversity_file = "active_dim_reduced_metrics/DIV2K_train_HR.csv"
    indexes, diversity = find_diverse_files(diversity_file, sample_size, pool_size, False)
    color = []
    for i in range(diversity.shape[0]):
        if i in indexes:
            color.append('green')
        else:
            color.append('red')
    adjacency_matrix = np.zeros([diversity.shape[0], diversity.shape[0]])
    # compute diversity matrix
    for i in range(diversity.shape[0]):
        for j in range(i + 1, diversity.shape[0]):
            u = np.array(diversity[i])
            v = np.array(diversity[j])
            u = u.reshape(1, u.shape[0])
            v = v.reshape(1, v.shape[0])
            temp = max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0])
            adjacency_matrix[i][j] = temp
            adjacency_matrix[j][i] = temp

    adjacency_matrix = np.rint(adjacency_matrix)
    adjacency_matrix = adjacency_matrix.astype(int)
    G = nx.from_numpy_matrix(adjacency_matrix, create_using=nx.MultiGraph)
    pos=nx.circular_layout(G) 
    nx.draw_networkx_nodes(G, pos, node_color=color, node_size=400)

    for (node1, node2, data_attr) in G.edges(data=True):
        width = data_attr['weight'] // 1000
        edge = (node1, node2)
        if width < 1:
            color = 'blue'
        elif width < 2:
            color = 'cyan'
        elif width < 3:
            color = 'magenta'
        elif width < 4:
            color = 'pink'
        else:
            color = 'black'
        width = width + 1
        nx.draw_networkx_edges(G, pos, edgelist=[edge], width=width, edge_color=color)

    plt.axis('off')
    plt.title('Visualizing k-means')
    plt.savefig(save_file)


if __name__ == "__main__":

    args = docopt(__doc__)
    diversity_file = Path(args["--diversity_file"]).resolve()
    save_file = Path(args["--save_file"]).resolve()
    create_graph(diversity_file, save_file)




