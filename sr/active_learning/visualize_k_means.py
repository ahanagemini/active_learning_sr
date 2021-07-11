import random
import numpy as np
import copy
from pathlib import Path
import os
from PIL import Image
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial.distance import directed_hausdorff
from k_means import find_diverse_files

def create_graph():
    """
    Function to compute 4 most similar and dissimilar images to image
    Returns: List of 100 images and their 4 most similar and dissimilar images
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

    print(adjacency_matrix)
    adjacency_matrix = np.rint(adjacency_matrix)
    adjacency_matrix = adjacency_matrix.astype(int)
    print(adjacency_matrix)
    G = nx.from_numpy_matrix(adjacency_matrix, create_using=nx.MultiGraph)
    pos=nx.circular_layout(G) 
    nx.draw_networkx_nodes(G, pos, node_color=color, node_size=400)

    print(G.nodes)
    for (node1, node2, data_attr) in G.edges(data=True):
        width = data_attr['weight']
        edge = (node1, node2)
        nx.draw_networkx_edges(G, pos, edgelist=[edge], width=width)


def save_images(sim_dissim_list):
    """
    Function to create and save images for visualizing dim_reductio.py quality
    Args: sim_dissim_list
    """
    filename = 'cmp_red_dim/' + (str(files_list[0]).split('/')[-1]).split('.')[0] + '.png'
    plt.imsave(filename, img_save, cmap="gray")

if __name__ == "__main__":
    create_graph()




