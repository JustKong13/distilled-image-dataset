import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
import numpy as np 
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.metrics import confusion_matrix

from utils import * 
from model import * 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ground Truth Dataset 
data = np.load('./CompressedDatasets/cifar/dataset.npz')
X_train = data['X_train']
y_train = data['y_train']
X_test = data['X_test']
y_test = data['y_test']


# Sanity check for reconstructing images
k = 1000
U, sigma, V = prepare_dataset(k)
print(f"U shape: {U.shape}, sigma shape: {sigma.shape}, V shape: {V.shape}")

reconstructed_img = reconstruct_img(U[1], sigma[:k], V[:k, :])

reconstructed_img = reconstructed_img.flatten()
U_test = (reconstructed_img @ torch.inverse(V)) @ torch.inverse(torch.diag(sigma))
new_img = U_test @ torch.diag(sigma) @ V
new_img = new_img.reshape(3, 32, 32)
new_img = new_img.permute(1, 2, 0)


def validate(model, test_loader): 
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_test_tensor, y_test_tensor in test_loader:
            X_test_tensor = X_test_tensor.to(device)
            y_test_tensor = y_test_tensor.to(device)

            y_test_tensor = y_test_tensor.flatten()
            outputs = model(X_test_tensor)
            _, predicted = outputs.max(1)
            correct += (predicted == y_test_tensor).sum().item()  # Accumulate correct predictions
            total += y_test_tensor.size(0)  # Accumulate total samples


    accuracy = 100. * correct / total  # Calculate overall accuracy
    # print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy


def train(model, train_loader, test_loader, optimizer, epochs, criterion, experiment_name = None): 
    train_accuracy = [] 
    val_accuracy = []

    # Training loop
    for epoch in tqdm(range(epochs)):
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0 

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device) 
            optimizer.zero_grad()
            outputs = model(inputs)
            labels = labels.squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        val_acc = validate(model, test_loader)
        val_accuracy.append(val_acc)

        train_accuracy.append(correct / total)


    # Save the model
    if experiment_name is not None:
        torch.save(model.state_dict(), f"./results/cifar-model/{experiment_name}.pth")
        print(f"Model saved as {experiment_name}.pth")
    else: 
        torch.save(model.state_dict(), f"./results/cifar-model/{experiment_name}.pth")
        print(f"Model saved as {experiment_name}.pth")

    return max(val_accuracy)



import warnings 
warnings.filterwarnings("ignore")

data = np.load('./CompressedDatasets/cifar/dataset.npz')
X_train = data['X_train']
y_train = data['y_train']
X_test = data['X_test']
y_test = data['y_test']
X_test = torch.tensor(X_test, dtype=torch.float32).permute(0, 3, 1, 2)

singular_values = [] 
accuracies = [] 

for k in range(200, 3072, 100):  
    U, sigma, V = prepare_dataset(k)
    U = torch.tensor(U, dtype=torch.float32, device=device)
    sigma = torch.tensor(sigma, dtype=torch.float32, device=device)
    V = torch.tensor(V, dtype=torch.float32, device=device)
    
    # Define training parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(sigma, V).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    batch_size = 64
    epochs = 25

    # Create DataLoader
    train_dataset = TensorDataset(torch.tensor(U, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(X_test, torch.tensor(y_test, dtype=torch.long))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Train the model
    experiment_name = f"model_{k}"
    acc = train(model, train_loader, test_loader, optimizer, epochs, criterion, experiment_name)
    
    accuracies.append(acc)
    singular_values.append(k)


import seaborn as sns 


sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.lineplot(x=singular_values, y=accuracies, markers='-o', color='b')

plt.title("Test Accuracy vs Number of Singular Values")
plt.xlabel("Number of Singular Values")
plt.ylabel("Test Accuracy (%)")
plt.savefig("test_accuracy_vs_singular_values.png")

np.savez("./CompressedDatasets/cifar/accuracy.npz", singular_values=singular_values, accuracies=accuracies)