import numpy as np 
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from typing import Tuple
import time 
import matplotlib.pyplot as plt
import os

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DatasetSVD: 
    """
    Class for the CIFAR-10 dataset. 
    Attributes:
        dataset_name (str): The name of the dataset. Currently only supports "CIFAR-10". 
        U (np.ndarray): The left singular vectors, shape (N, N)
        sigma (np.ndarray): The singular values, shape (N, D)
        Vt (np.ndarray): The right singular vectors, shape (D, D)
        X_train (np.ndarray): The training set, shape (N, D)
        y_train (np.ndarray): The labels for the training set, shape (N,)
        X_test (np.ndarray): The test set, shape (N, D)
        y_test (np.ndarray): The labels for the test set, shape (N,)
    """
    def __init__(self, 
                 fft: bool = False,  
                 dataset:str = "CIFAR-10", 
                 path_to_SVD_dataset:str = None, 
                 load_k_svd:int = None): 
        """
        Initializes the CIFAR-10 dataset and computes the SVD for the training set. If path is 
        provided, it will load the dataset from the path.
        Params: 
            dataset (str): The name of the dataset. Currently only supports "CIFAR-10". 
            path (str): The path to the dataset, and corresponding attributes. 
        """
        self.dataset_name = dataset
        self.allow_fft = fft
        self.path = path_to_SVD_dataset


        if self.dataset_name == "CIFAR-10" and self.path is None:
            self.init_cifar()
        # elif self.dataset_name == "CIFAR-10" and path != "":
        #     self.load(path)

        self.U: np.ndarray
        self.sigma: np.ndarray
        self.Vt: np.ndarray
        self.X_train: np.ndarray
        self.y_train: np.ndarray
        self.X_test: np.ndarray
        self.y_test: np.ndarray

        
    def init_cifar(self): 
        """
        Initialize the CIFAR-10 dataset and computes the SVD for the training set. 
        """
        from tensorflow.keras.datasets import cifar10   
        # Load CIFAR-10 dataset
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        X_train = X_train.astype(np.float32) / 255.0  # Normalize to [0,1]
        X_test = X_test.astype(np.float32) / 255.0  # Normalize to [0,1]

        if self.allow_fft: 
            X_train = fft2(X_train, axes=(1,2))
            X_train = fftshift(X_train, axes=(1,2))
            X_train = np.abs(X_train) # or maybe np.real? 

        U, sigma, Vt = self.SVD_dataset(X_train)

        self.U = U 
        self.sigma = sigma 
        self.Vt = Vt 
        self.X_train = X_train 
        self.y_train = y_train 
        self.X_test = X_test 
        self.y_test = y_test


    def unpickle(self, file) -> dict:
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict
    
    def save(self, path:str = None, k:int = 3072) -> None: 
        """
        Saves the SVD of the dataset to a file. 
        Params: 
            path (str): The path to save the SVD to. If None, saves to the default path.
        """
        if path is None: 
            path = "checkpoints/svd_cifar10.pth"

        if self.allow_fft: 
            path = path.replace(".pth", "_fft.pth")

        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the SVD to a file
        np.savez(path, U=self.U[:, :k], sigma=self.sigma, Vt=self.Vt)
        print(f"Saved SVD to {path}")


    def load(self, path:str = None) -> None: 
        """
        Loads the SVD of the dataset from a file. 
        Params: 
            path (str): The path to load the SVD from. If None, loads from the default path.
        """
        assert path is not None, "Path to SVD file must be provided"
        
        dataset_path = path + "_dataset.npz"

        data = np.load(path)
        self.U = data['U']
        self.sigma = data['sigma']
        self.Vt = data['Vt']
        print(f"Loaded SVD from {path}")

        dataset = np.load(dataset_path)
        self.X_train = dataset['X_train']
        self.y_train = dataset['y_train']
        self.X_test = dataset['X_test']
        self.y_test = dataset['y_test']
        print(f"Loaded dataset from {dataset_path}")


    def get_compression_ratio(self, k: int = 3072) -> float:
        """
        Returns the compression ratio for the dataset. 
        Params: 
            k (int): The number of singular values to use for reconstruction
        Returns:
            compression_ratio (float): The compression ratio
        """
        # Compression Ratio = (Original Size) / (Compressed Size)
        original_size = self.X_train.nbytes
        compressed_size = (self.U[:k].nbytes + self.sigma[:k].nbytes + self.Vt[:k].nbytes)
        compression_ratio = original_size / compressed_size
        return compression_ratio


    def SVD_dataset(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: 
        """
        Returns the Singular Value Decomposition for the entire dataset 
        Params: 
            data (np.ndarray): The dataset, shape (N, D) where N is the number of samples and D is 
            the dimensionality of the data
        Returns:
            U (np.ndarray): The left singular vectors, shape (N, N)
            S (np.ndarray): The singular values, shape (N, D)
            V (np.ndarray): The right singular vectors, shape (D, D)
        """
        
        # Preprocessing & Normalization
        data = data.reshape(data.shape[0], -1) 
        # data = data - data.mean(axis=0)
        # data = data / data.std(axis=0)
        
        # Singular Value Decomposition 
        U, sigma, Vt = np.linalg.svd(data, full_matrices=False)
        return U, sigma, Vt


    def reconstruct_image_CIFAR(self, U: np.ndarray, k: int = 3072) -> np.ndarray: 
        """
        Reconstructs the image using the first k singular values and vectors
        Params: 
            U (np.ndarray): A left singular vector, shape (N,)
            sigma (np.ndarray): The singular values, shape (N, D)
            Vt (np.ndarray): The right singular vectors, shape (D, D)
            k (int): The number of singular values to use for reconstruction
        Returns:
            reconstructed_image (np.ndarray): The reconstructed image, shape (N, N, D)
        """
        # Reconstruct the image using the first k singular values and vectors
        # A_k = U_k Σ_k V_k^T 
        reconstructed_image = U[:k].T @ (np.diag(self.sigma[:k]).T @  self.Vt[:k, :])
        reconstructed_image = reconstructed_image.reshape(32, 32, 3)
        reconstructed_image = np.clip(reconstructed_image, 0, 1)

        if self.allow_fft: 
            reconstructed_image = ifftshift(reconstructed_image, axes=(1,2))
            reconstructed_image = ifft2(reconstructed_image, axes=(1,2))
            reconstructed_image = np.abs(reconstructed_image)
            reconstructed_image = np.clip(reconstructed_image, 0, 1)

        return reconstructed_image


    def compare_reconstruction(self, i: int, k: int = 3072) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: 
        """
        Visualizes the reconstruciton of the i-th image in the dataset using the first k singular 
        values. 
        Args: 
            i (int): The index of the image to reconstruct
            k (int): The number of singular values to use for reconstruction
        Returns: 
            Original Image, Fully Reconstructed Image, Partially Reconstructed Image 
        """
        d1 = self.X_train.shape[1]
        d2 = self.X_train.shape[2]
        U_i = self.U[i]

        # Original Image (Ground Truth)
        original = self.X_train[i] 
        original = original.reshape(d1, d2, 3)
        original = np.clip(original, 0, 1)

        # Fully Reconstructed Image 
        fully_reconstructed = self.reconstruct_image_CIFAR(U_i,)
        fully_reconstructed = np.clip(fully_reconstructed, 0, 1)
        
        # Partially Reconstructed Image 
        start = time.perf_counter() 
        partially_reconstructed = self.reconstruct_image_CIFAR(U_i, k=k)
        partially_reconstructed = np.clip(partially_reconstructed, 0, 1)
        end = time.perf_counter() 
        print(f"Reconstruction time for k={k}: {end - start:.4f} seconds")

        # Plot the images 
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(original)
        ax[0].set_title("Original Image")
        ax[1].imshow(fully_reconstructed)
        ax[1].set_title("Fully Reconstructed Image")
        ax[2].imshow(partially_reconstructed)
        ax[2].set_title(f"Partially Reconstructed Image (k={k})")

        return original, fully_reconstructed, partially_reconstructed


    def plot_singular_values(self): 
        """
        Plots the singular values in a bar chart. 
        """
        plt.plot(self.sigma, 'o-')
        plt.yscale('log')
        plt.title("CIFAR 10: Singular Value")
        plt.xlabel("Index")
        plt.ylabel("Singular Value")
        plt.grid(True)


    def add_image_train(self, image: np.ndarray, label: int) -> None: 
        """
        Adds an image to the dataset. This function adds the image to X_train, y_train, and computes the row contribution to U. 
        Params: 
            image (np.ndarray): The image to add, shape (32, 32, 3)
            label (int): The label for the image
        """

        assert image.shape == (32, 32, 3), "Image must be of shape (32, 32, 3)"
        assert label in range(10), "Label must be between 0 and 9"
        
        # Add the image to the training set
        self.X_train = np.append(self.X_train, [image], axis=0)
        self.y_train = np.append(self.y_train, [label])

        image = image.flatten() 
        new_U = (image @ np.linalg.inv(self.Vt)) @ np.linalg.inv(np.diag(self.sigma))
        self.U = np.append(self.U, [new_U], axis=0)
        return new_U 
    
    def compute_U_for_image(self, image: np.ndarray) -> np.ndarray: 
        """
        Computes the U matrix for a given image using SVD.
        Params:
            img (np.ndarray): The input image.
        Returns:
            np.ndarray: The U matrix row contribution for the image.
        """
        assert image.shape == (32, 32, 3), "Image must be of shape (32, 32, 3)"
        
        # Flatten the image
        image = image.flatten() 
        new_U = (image @ np.linalg.inv(self.Vt)) @ np.linalg.inv(np.diag(self.sigma))
        return new_U