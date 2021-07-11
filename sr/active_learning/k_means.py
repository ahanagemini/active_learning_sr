"""Usage:    k_means.py --diversity_file=diversity_file --sample_size=size
             k_means.py --help | -help | -h

Use shte dim reduced values in order to compute the k diverse files
Arguments:
  diversity_file      a file that contains diversity values for the samples
  sample_size         the number of samples to select

Options:
  -h --help -h
"""
import time
import statistics
import random
from docopt import docopt
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import KDTree

def find_diverse_files(diversity_file, sample_size, pool_size=None, save_files=True):
    '''
    Function to read the diversity file and return k-means NN file names
    Args: diversity_file: diversity file name
          sample_size: sample size to be selected
    Returns: list of diverse files
    '''

    # Read the dimension reduced data from the file
    diversity = pd.read_csv(diversity_file, header=None,
                            names=['filename'] + [str(i) for i in range(0, 128)])
    # Sort by filename
    if pool_size is not None:
        diversity = diversity.sample(n=pool_size)
    sorted_diversity = diversity.sort_values(by='filename')
    filenames = list(sorted_diversity.loc[:, 'filename'])
    diversity_values = sorted_diversity.drop('filename', axis='columns')

    # Use k-means to identify the k-centers
    ndims = diversity_values.to_numpy()
    k = sample_size
    kmeans = KMeans(n_clusters=k, init='random', random_state=0).fit(ndims)
    k_list = kmeans.cluster_centers_

    # identify the closest NN point for each k-center
    kdtree = KDTree(ndims)
    _, indexes = kdtree.query(k_list)
    file_list = [filenames[i] for i in indexes]
    print(len(file_list))
    file_set = list(set(file_list))
    print(len(file_set))

    if save_files:
    # write selected filenames to a file
        MyFile = open(f'{sample_size}_diverse.txt','w')
        result = [file_set[i]+'\n' for i in range(len(file_set))]
        MyFile.writelines(result)
        MyFile.close()

    return list(set(indexes)), ndims


if __name__ == "__main__":
    args = docopt(__doc__)
    diversity_file = Path(args["--diversity_file"]).resolve()
    sample_size = int(args["--sample_size"])
    start = time.time()
    indexes, diversity = find_diverse_files(diversity_file, sample_size)
