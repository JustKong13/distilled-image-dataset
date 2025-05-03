import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
import numpy as np 
from tqdm import tqdm 
from torch.utils.data import DataLoader, Dataset, TensorDataset 

class SimpleCNN(nn.Module):
    def __init__(self, sigma, Vt):
        super(SimpleCNN, self).__init__()
        self.sigma = sigma 
        self.Vt = Vt 

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def reconstruct_img(self, x) -> torch.Tensor: 
        k = x.shape[1] # number of singular values 
        x = x @ (torch.diag(self.sigma[:k]) @  self.Vt[:k, :])
        x = x.reshape(x.shape[0], 32, 32, 3)
        x = x.permute(0, 3, 1, 2)
        x = torch.clip(x, 0, 1)
        return x # [N, 3, 32, 32]

    def forward(self, x):
        """
        Forward pass through the network.
        """
        # Apply SVD transformation
        if len(x.shape) == 2:
            x = self.reconstruct_img(x)

        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x