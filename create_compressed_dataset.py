from preprocess_SVD import * 
import time 


dataset = DatasetSVD(fft=False)

def create_dataset(singular_values): 
    original, fully_reconstructed, partially_reconstructed = dataset.compare_reconstruction(1, k=singular_values)
    dir = f"./CompressedDatasets/cifar/cifar10_{singular_values}"
    dataset.save(path=dir, k=singular_values)

X_train, y_train, X_test, y_test = dataset.X_train, dataset.y_train, dataset.X_test, dataset.y_test

np.savez(f"./CompressedDatasets/cifar/dataset.npz", X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

for i in range(200, 3200, 100): 
    print(f"Creating dataset with {i} singular values")
    start = time.time()
    create_dataset(i)
    end = time.time()
    print(f"Time taken: {end - start} seconds")
