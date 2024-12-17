import numpy as np
import argparse
import torch
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from DataLoader import load_data, train_valid_split, CustomDataset
from ImageUtils import data_stats, visualize, visualize_pca
from CNN import CNN, train_model, validate_model, test_model, load_and_test
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from RandomForest import rf_train_model, rf_test_model, rf_test_best_model
from ResNet import ResNet
from sklearn.manifold import TSNE

# Class names corresponding to CIFAR-10 labels
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

os.makedirs('./models', exist_ok=True)

# ------- DATA STATS -------- #

x_train, y_train, x_test, y_test = load_data('data')
x_train, y_train, x_valid, y_valid = train_valid_split(x_train, y_train)
print("Shape of X_train:", x_train.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of X_val:", x_valid.shape)
print("Shape of y_val:", y_valid.shape)
print("Shape of X_test:", x_test.shape)
print("Shape of y_test:", y_test.shape)

# Assuming y_train is the list or array of labels for CIFAR-10
class_distribution = data_stats(y_train)
print(class_distribution)


# --------------- EDA ----------------- #

# Visualizing Training Data
#visualize(x_train,y_train)

# Applying PCA and visualizing 
# visualization 1 - 3D Visualization with 3 principal components
# Visualization 2 - 2D tsne visualization
# Visualization 3 - Reconsrtucted Images using 250 principal components

# Flatten the images to (n_samples, n_features) for PCA (3072 features per image)
x_train_flat = x_train.reshape(x_train.shape[0], -1)  # Shape: (40000, 3072)
x_valid_flat = x_valid.reshape(x_valid.shape[0], -1)
x_test_flat = x_test.reshape(x_test.shape[0], -1)

# Initialize PCA with 250 components
pca = PCA(n_components=250)

# Fit and transform the flattened images
x_train_pca = pca.fit_transform(x_train_flat)  # Shape: (40000, 50)
x_test_pca = pca.transform(x_test_flat)
x_valid_pca = pca.transform(x_valid_flat)

# Project the PCA-reduced data back to the original space (inverse transform)
x_train_pca_back = pca.inverse_transform(x_train_pca)  # Shape: (40000, 3072)

# Reshape back to (n_samples, 3, 32, 32)
x_train_pca_back = x_train_pca_back.reshape(x_train.shape)  # Shape: (40000, 3, 32, 32)

#visualize_pca(x_train_pca, x_train_pca_back, y_train)

#tsne visualization
#tsne = TSNE(n_components=2, perplexity=40, n_iter=2000, random_state=42)
#X_tsne = tsne.fit_transform(x_train_pca)

#plt.figure(figsize=(12, 8))
#scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_train, cmap='tab10', s=10)
#plt.colorbar(scatter, ticks=range(10), label='Class Labels')
#plt.title('t-SNE plot of CIFAR-10 Dataset with Improved Separation')
#plt.xlabel('t-SNE Component 1')
#plt.ylabel('t-SNE Component 2')
#plt.show()

# ------- DATA PREPROCESSING -------- #

#Transform for normalization and conversion to tensor
transform = transforms.Compose([
    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
    transforms.ToTensor(),  # Convert to tensor and scale to [0, 1]
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])

test_transform = transforms.Compose([
    transforms.ToTensor(),  # Convert to tensor and scale to [0, 1]
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])

# Create PyTorch Datasets and DataLoaders for original Data
train_dataset = CustomDataset(x_train, y_train, transform=transform)
valid_dataset = CustomDataset(x_valid, y_valid, transform=transform)
test_dataset = CustomDataset(x_test, y_test, transform=test_transform)


# Also experimented with Batch_size 32, 64 and 128 while Training the models
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)



# ----------- EXPERIMENTATION AND MODEL TRAINING ------------- #

######---------   MODEL 1: Random Forest Classifier  -----------######

#Training and testing the model 
# model = rf_train_model(x_train_pca, y_train, x_valid_pca, y_valid)
# y_pred, accuracy = rf_test_model(model, x_test_pca, y_test)

# Using the best model on the testing data
#rf_test_best_model(x_test_pca, y_test)



######---------   MODEL 2: CNN   ------------######

# Experimenting with different Hyperparameters

#filter_sizes = [3, 5]
#num_filters_list = [32, 64]
#activation_functions = ['relu', 'sigmoid', 'tanh']

#best_accuracy = 0
#best_model = None
#best_hyperparameters = {}
# model_save_path_CNN_Train = './models/CNN_TrainData_best_model.pth'

# Iterate over all combinations of hyperparameters for Training and Validation and save the model with best Hyperparameters

#for filter_size in filter_sizes:
#    for num_filters in num_filters_list:
#        for activation in activation_functions:
#            print(f'\nExperimenting with filter_size={filter_size}, num_filters={num_filters}, activation={activation}')
#            model = CNN(filter_size=filter_size, num_filters=num_filters, activation=activation)
#            train_model(model, train_loader, valid_loader)
             # Get accuracy on validation set
#            _, accuracy = validate_model(model, valid_loader)
#            if accuracy > best_accuracy:
#                best_accuracy = accuracy
#                best_model = model
#                best_hyperparameters = {'filter_size': filter_size, 'num_filters': num_filters, 'activation': activation}

                 # Save the best model
#                torch.save({
#                    'model_state_dict': best_model.state_dict(),
#                    'hyperparameters': best_hyperparameters,
#                    'accuracy': best_accuracy
#                }, model_save_path_CNN_Train)
#                print(f"New best model saved with accuracy: {best_accuracy:.2f}%")
#print(f'\nBest hyperparameters: {best_hyperparameters}')
#print(f'Best accuracy on validation set: {best_accuracy:.2f}%')



# Perform inference on test dataset using the best model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the saved model
#checkpoint = torch.load("CNN_64_3_64_Relu.pth")
#loaded_model = CNN(
#     filter_size=checkpoint['hyperparameters']['filter_size'],
#     num_filters=checkpoint['hyperparameters']['num_filters'],
#     activation=checkpoint['hyperparameters']['activation']
#)
#loaded_model.load_state_dict(checkpoint['model_state_dict'])

#loaded_model.to(device)
#print(loaded_model)

# Ensure the model is in evaluation mode
#loaded_model.eval()

# Print the saved model's hyperparameters
#print(f"Loaded model with best accuracy")
#print(f"Hyperparameters: {checkpoint['hyperparameters']}")
# Use the loaded model to evaluate on test set
#load_and_test(loaded_model, device, test_loader)




######-----------          MODEL 3: Resnet           -----------######


# Training the Model and saving the best performing model
#resnet = ResNet(device, 0.001)
#resnet.train_model(train_loader, valid_loader, 20)


# Loading the saved model and getting the predictions on the test data

checkpoint = torch.load("models/ResNet_best_model.pth", map_location=device)
loaded_model = ResNet(device,
    checkpoint['hyperparameters']['learning_rate'],
    checkpoint['hyperparameters']['num_filters'],
    checkpoint['hyperparameters']['kernel_size'],
    checkpoint['hyperparameters']['optimizer'],
    checkpoint['hyperparameters']['activation']
)
loaded_model.model.load_state_dict(checkpoint['model_state_dict'])

loaded_model.to(device)
#print(loaded_model)

# Ensure the model is in evaluation mode
loaded_model.eval()
predictions = loaded_model.test_model(test_loader)

###### ------ The above Loaded ResNet model is the best performing model ------######
# Print and Save the predictions into a csv 
print(predictions)
np.savetxt('predictions.csv', predictions, delimiter=",", fmt='%d')





