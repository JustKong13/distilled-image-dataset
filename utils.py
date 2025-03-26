import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from typing import Tuple


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def SVD_dataset(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: 
    """
    Returns the Singular Value Decomposition for the entire dataset 
    Params: 
        data (np.ndarray): The dataset, shape (N, D) where N is the number of samples and D is the dimensionality of the data
    Returns:
        U (np.ndarray): The left singular vectors, shape (N, N)
        S (np.ndarray): The singular values, shape (N, D)
        V (np.ndarray): The right singular vectors, shape (D, D)
    """
    
    # Preprocessing & Normalization
    data = data.reshape(data.shape[0], -1) 
    data = data - data.mean(axis=0)
    data = data / data.std(axis=0)
    
    # Singular Value Decomposition 
    U, sigma, Vt = np.linalg.svd(data)
    return U, sigma, Vt