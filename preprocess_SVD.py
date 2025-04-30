import numpy as np 
from typing import Tuple
import time 
import matplotlib.pyplot as plt

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
    def __init__(self, dataset:str = "CIFAR-10", path:str = None): 
        """
        Initializes the CIFAR-10 dataset and computes the SVD for the training set. If path is 
        provided, it will load the dataset from the path.
        Params: 
            dataset (str): The name of the dataset. Currently only supports "CIFAR-10". 
            path (str): The path to the dataset, and corresponding attributes. 
        """
        self.dataset_name = dataset

        if self.dataset_name == "CIFAR-10" and path is None:
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
    
    def save(self, path:str = None) -> None: 
        """
        Saves the SVD of the dataset to a file. 
        Params: 
            path (str): The path to save the SVD to. If None, saves to the default path.
        """
        if path == None: 
            path = "checkpoints/svd_cifar10.pth"
            dataset_path = "checkpoints/cifar10_dataset.npz"

        np.savez(path, U=self.U, sigma=self.sigma, Vt=self.Vt)
        print(f"Saved SVD to {path}")

        np.savez(dataset_path, X_train=self.X_train, y_train=self.y_train, X_test=self.X_test, y_test=self.y_test)
        print(f"Saved dataset to {dataset_path}")


    def load(self, path:str = None) -> None: 
        """
        Loads the SVD of the dataset from a file. 
        Params: 
            path (str): The path to load the SVD from. If None, loads from the default path.
        """
        if path == None: 
            path = "checkpoints/svd_cifar10.pth"
            dataset_path = "checkpoints/cifar10_dataset.npz"

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

        return reconstructed_image


    def compare_reconstruction(self, i: int, k: int = 3072) -> None : 
        """
        Visualizes the reconstruciton of the i-th image in the dataset using the first k singular 
        values. 
        Args: 
            i (int): The index of the image to reconstruct
            k (int): The number of singular values to use for reconstruction
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