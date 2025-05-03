import numpy as np 
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch
import matplotlib.pyplot as plt 


def prepare_dataset(singular_values: int) -> TensorDataset: 
    """
    Prepares the CIFAR-10 dataset for training and testing.
    Args:
        singular_values (int): Number of singular values to use for SVD.
    Returns:
        U, sigma, Vt (torch.Tensor): The left singular vectors, singular values, and right singular vectors.
    """
    file_name = f"./CompressedDatasets/cifar/cifar10_{singular_values}.npz"
    data = np.load(file_name) 
    U = data["U"]
    sigma = data["sigma"]
    Vt = data["Vt"]

    return torch.tensor(U), torch.tensor(sigma), torch.tensor(Vt) 


def reconstruct_img(x, sigma, Vt) -> torch.Tensor: 
    x = x @ (torch.diag(sigma) @  Vt)
    x = x.reshape(32, 32, 3)
    plt.imshow(x)
    x = x.permute(2, 0, 1)
    x = torch.clip(x, 0, 1)
    return x # [N, 3, 32, 32]


def compute_U_for_image(img:np.ndarray, sigma, Vt) -> np.ndarray: 
    """
    Computes the U matrix for a given image using SVD.
    Args:
        img (np.ndarray): The input image.
        sigma (np.ndarray): The singular values.
        Vt (np.ndarray): The right singular vectors.
    Returns:
        np.ndarray: The U matrix for the image.
    """
    raise NotImplementedError