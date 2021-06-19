"""Usage:    utility_sort.py --utility_file=utility_file
             utility_sort.py --help | -help | -h

Use shte utility value sand the diversity values in order to compute the best active learning samples
Arguments:
  utility_file      a file that contains utility values for the samples

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

def read_utility(utility_file):
    '''
    Function to read the utility file and return a list containing utility values
    Args: utility file name
    Returns: list of utilities sorted by file name
    '''
    utility = pd.read_csv(utility_file)
    sorted_utility = utility.sort_values(by='SR_L1',ascending=False)
    print(sorted_utility, sorted_utility.dtypes)
    return sorted_utility['filename'].to_list()


if __name__ == "__main__":
    args = docopt(__doc__)
    utility_file = Path(args["--utility_file"]).resolve()
    start = time.time()
    # Read and sort based on L1 loss utility
    files = read_utility(utility_file)

    # Open and write sorted data to file
    MyFile = open(f'utility_sorted.txt','w')
    result = [files[i]+'\n' for i in range(len(files))]
    MyFile.writelines(result)
    MyFile.close()
