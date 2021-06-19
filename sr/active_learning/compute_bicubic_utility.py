"""Usage:   compute_bicubic_utility.py --img_dir=img_dir
            compute_bicubic_utility.py --help | -help | -h

Perform dimension reduction of images in a directory
Arguments:
  img_dir   directory of training images for dimension reduction

Options:
  -h --help -h
"""

import time
from docopt import docopt
from pathlib import Path
import numpy as np
from scipy.fftpack import fft, dct
from sklearn.decomposition import IncrementalPCA
from skimage.transform import resize
import math
import csv


def compute_utility(img_dir):
    '''
    Function to read the images in a directory and perform utility computation
    Use downsampling and upsampling (bicubic) to compute the 
    L1 loss between original and upsampled image
    Args: img_dir: image directory
    Returns: list of utility for each image
    '''
    utility_list = []
    imglist = list(img_dir.rglob("*.npz"))
    imglist = sorted(imglist)
    read_time = 0.0
    compute_time = 0.0
    for img_name in imglist:
        # read image
        start = time.time()
        image = np.load(img_name)
        image = image.f.arr_0
        read_time = read_time + time.time() - start
        # rescale to 64 * 64
        start = time.time()
        size = image.shape[0] // 4
        image_small = resize(image, (size, size), order=3, preserve_range=True)
        image_lr =  resize(image_small, (image.shape[0], image.shape[1]), order=3, preserve_range=True)
        utility = np.mean(np.abs(image - image_lr))
        utility_list.append(utility)
        compute_time = compute_time + time.time() - start

    print(f"Read_time {read_time}, Compute time {compute_time}")
    return utility_list, imglist 

def write_utility(img_dir, utility_list, imglist):

    '''
    Function to write utility to file
    Args: 
        img_dir: image directory
        utility_list: the list to write to file
        imglist: file names for each item in the list
    '''
    write_list = [['filename', 'SR_L1']]
    [write_list.append([str(imglist[i]).split('/')[-1], utility_list[i]]) for i in range(len(utility_list))]
    
    active_file = str(img_dir).split('/')[-1]
    with open(f"active_utility_metrics/{active_file}.csv", 'w') as f:
        # using csv.writer method from CSV package
        write = csv.writer(f)
        write.writerows(write_list)



if __name__ == "__main__":
    args = docopt(__doc__)
    img_dir = Path(args["--img_dir"]).resolve()
    start = time.time()
    utility_list, imglist = compute_utility(img_dir)
    write_utility(img_dir, utility_list, imglist)
    print(f"Total time {time.time() - start}")

