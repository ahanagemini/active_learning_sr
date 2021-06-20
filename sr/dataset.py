"""
Dataset file
"""
from pathlib import Path
import json
import random
import os

# from matplotlib import pyplot as plt
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import torch.nn.functional as F
import scipy.ndimage
import numpy as np
from cutter import loader
from skimage.transform import resize


class SrDataset(Dataset):
    """Dataset class for loading large amount of image arrays data"""

    def __init__(self, root_dir, lognorm=False, test=False, hr=True):
        """
        Args:
            root_dir (string): Directory with all the images.
            lognorm: True if we ar eusing log normalization
            test: True only for test dataset
            hr: Input is hr image, lr is computed, then True
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = Path(root_dir).expanduser().resolve().absolute()
        self.datalist = list(self.root_dir.rglob("*.npz"))
        self.lognorm = lognorm
        self.test = test
        self.hr = hr
        self.statlist = []
        for fname in self.datalist:
            file_path = Path(fname)
            stat_file = json.load(open(str(file_path.parent / "stats.json")))
            self.statlist.append(stat_file)
        print("Total number of data elements found = ", len(self.datalist))

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        img_name = Path(self.datalist[idx])
        filename = os.path.basename(img_name)
        filename = filename.split('.')[0]
        stats = self.statlist[idx]
        if self.hr:
            hr_image = loader(img_name)
        if not self.test:
            if stats["std"] <= 0.001:
                stats["std"] = 1
            hr_image = Normalize()(hr_image, stats)
        
        if self.hr:
            lr_image = scipy.ndimage.zoom(scipy.ndimage.zoom(hr_image, 0.25), 4.0)
        else:
            lr_image =  loader(img_name)
            hr_image = np.zeros_like(lr_image)
        if not self.test:
            sample = {"lr": lr_image, "lr_un": lr_image, "hr": hr_image, "stats": stats, "file": filename}
            transforms = Compose(
                [Rotate(), Transpose(), HorizontalFlip(), VerticalFlip(),
                    Reshape(), ToFloatTensor()]
            )
            for i, trans in enumerate([transforms]):
                sample = trans(sample)
        else:
            lr_unorm = lr_image.copy()
            if stats["std"] <= 0.001:
                stats["std"] = 1
            lr_image = Normalize()(lr_image, stats)
            sample = {"lr": lr_image, "lr_un": lr_unorm, "hr": hr_image, "stats": stats, "file": filename}
            transforms = Compose(
                [Reshape(), ToFloatTensor()]
            )
            sample = transforms(sample)
        return sample


"""
if __name__ == "__main__":
    face_dataset = SrDataset(root_dir='../data')

    fig = plt.figure()

    for i in range(len(face_dataset)):
        sample = face_dataset[i]

        print(i, sample['hr'].shape, sample['stats'])

        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')
        plt.imshow(sample['lr'])

        if i == 3:
            plt.show()
            break
"""


class Rotate:
    """Rotate class rotates image array"""

    def __call__(self, sample):
        """

        Parameters
        ----------
        sample: dictionary containing lr, hr and stats

        Returns
        -------
        sample: dictionary containing transformed lr and transformed hr
        """
        for i in range(random.randint(0, 3)):
            sample["hr"] = np.rot90(sample["hr"])
            sample["lr"] = np.rot90(sample["lr"])

        return sample


class ToFloatTensor:
    """This class is for converting the image array to Float Tensor"""

    def __call__(self, sample):
        """
        Parameters
        ----------
        sample: dictionary containing lr, hr and stats

        Returns
        -------
        sample: dictionary containing transformed lr and transformed hr
        """
        sample["hr"] = np.ascontiguousarray(sample["hr"])
        sample["lr"] = np.ascontiguousarray(sample["lr"])
        sample["hr"] = torch.tensor(sample["hr"], dtype=torch.float32)
        sample["lr"] = torch.tensor(sample["lr"], dtype=torch.float32)
        return sample


class Transpose:
    """Transpose class calculates the transpose of the matrix"""

    def __call__(self, sample):
        """

        Parameters
        ----------
        sample: dictionary containing lr, hr and stats

        Returns
        -------
        sample: dictionary containing transformed lr and transformed hr
        """
        if random.randint(1, 10) > 5:
            sample["hr"] = np.transpose(sample["hr"])
            sample["lr"] = np.transpose(sample["lr"])

        return sample

class VerticalFlip:
    """VerticalFlip class to probailistically return vertical flip of the matrix"""

    def __call__(self, sample):
        """

        Parameters
        ----------
        sample: dictionary containing lr, hr and stats

        Returns
        -------
        sample: dictionary containing transformed lr and transformed hr
        """
        if random.randint(1, 10) > 5:
            sample["hr"] = np.flipud(sample["hr"])
            sample["lr"] = np.flipud(sample["lr"])

        return sample

class HorizontalFlip:
    """HorizontalFlip class to probailistically return horizontal flip of the matrix"""

    def __call__(self, sample):
        """

        Parameters
        ----------
        sample: dictionary containing lr, hr and stats

        Returns
        -------
        sample: dictionary containing transformed lr and transformed hr
        """
        if random.randint(1, 10) > 5:
            sample["hr"] = np.fliplr(sample["hr"])
            sample["lr"] = np.fliplr(sample["lr"])

        return sample


class Reshape:
    """Reshaping tensors"""

    def __call__(self, sample):
        """

        Parameters
        ----------
        sample: dictionary containing lr, hr and stats

        Returns
        -------
        sample: dictionary containing reshaped lr and reshaped hr
        """
        width = sample["hr"].shape[-1]
        sample["hr"] = np.reshape(sample["hr"], (1, -1, width))
        sample["lr"] = np.reshape(sample["lr"], (1, -1, width))
        sample["lr_un"] = np.reshape(sample["lr_un"], (1, -1, width))
        return sample


class Normalize:
    """Normalizing the high resolution image using mean and standard deviation"""

    def __call__(self, hr_image, stats):
        """

        Parameters
        ----------
        hr_image: high resolution image
        stats: containing mean and standard deviation

        Returns
        -------
        hr_image: returns normalized hr image
        """
        return (hr_image - stats["mean"]) / stats["std"]
