#import torch
import numpy as np
from collections import Counter
from PIL import Image
from matplotlib import pyplot as plt

# Class names corresponding to CIFAR-10 labels
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def data_stats(y_train):
  class_counts = Counter(y_train)
  class_counts_named = {class_names[label]: count for label, count in class_counts.items()}
  return class_counts_named

def visualize(x_train, y_train):
    # Creating a dictionary to store one image from each class
    class_images = {}

    # Looping through the training data and picking the first image from each class
    for i in range(len(x_train)):
      label = y_train[i].item()  
      if label not in class_images:
        class_images[label] = x_train[i] 
      # WHen we have images from each class, we stop
      if len(class_images) == 10: 
        break

    # Visualize one image from each class
    for label, image_tensor in class_images.items():
      #print(class_images)
      class_name = class_names[label]  
      title = f'Train-Image-{class_name}'  
      image = np.transpose(image_tensor.reshape((3, 32, 32)), [1, 2, 0])
      plt.imshow(image)
      plt.savefig(title)

def visualize_pca(x_train_pca,x_train_pca_back, y_train):
    # Create a dictionary to store one image from each class after PCA
    class_images = {}

    # Loop through the PCA-reconstructed data and pick the first image from each class
    for i in range(len(x_train_pca_back)):
      label = y_train[i].item()  # Get the class label of the current image
        
      if label not in class_images:
        class_images[label] = x_train_pca_back[i]  # Store the first image from each class
        
      # When we have images from each class, stop
      if len(class_images) == 10:  
        break

    # Visualize one image from each class after PCA
    for label, image_tensor in class_images.items():
      class_name = class_names[label]  # Get the class name from the class_names list
      title = f'PCA-Image-{class_name}'  # Title for the plot
        
      # The image_tensor is of shape (3, 32, 32), so transpose to (32, 32, 3) for visualization
      image = np.transpose(image_tensor.reshape((3, 32, 32)), [1, 2, 0])  # Shape (32, 32, 3)
        
      # Plot and save the PCA-reconstructed image for this class
      plt.imshow(image.astype(np.uint8))  # Convert to uint8 for proper image visualization
      plt.title(class_name)  # Add the class name as the title
      plt.axis('off')  # Hide axes for clarity
      plt.savefig(f'{title}.png')  # Save the image as a file
      plt.show()  # Display the image

    # Visualize 3D PCA representation
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot
    scatter = ax.scatter(
      x_train_pca[:, 0],  # First principal component
      x_train_pca[:, 1],  # Second principal component
      x_train_pca[:, 2],  # Third principal component
      c=y_train,           # Use labels for color
      cmap=plt.get_cmap('viridis'),
      marker='o',
      alpha=0.7
    )

    # Set labels and title
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    ax.set_title('3D PCA Visualization of Cifar10')

    # Add colorbar
    cbar = fig.colorbar(scatter)
    cbar.set_label('Class Label')
    plt.savefig(f'pca-3D.png')
    plt.show()



