import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# Class names corresponding to CIFAR-10 labels
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Define CNN model with configurable hyperparameters
class CNN(nn.Module):
  
  def __init__(self, filter_size=3, num_filters=32, activation='relu'):
    super(CNN, self).__init__()

    # Setting the activation functions
    if activation == 'relu':
      self.activation = nn.ReLU()
    elif activation == 'sigmoid':
      self.activation = nn.Sigmoid()
    elif activation == 'tanh':
      self.activation = nn.Tanh()

    # Layers
    self.conv1 = nn.Conv2d(3, num_filters, kernel_size=filter_size, padding=(filter_size - 1) // 2)
    self.bn1 = nn.BatchNorm2d(num_filters)
    self.conv2 = nn.Conv2d(num_filters, 128, kernel_size=filter_size, padding=(filter_size - 1) // 2)
    self.bn2 = nn.BatchNorm2d(128)
    self.conv3 = nn.Conv2d(128, 256, kernel_size=filter_size, padding=(filter_size - 1) // 2)
    self.bn3 = nn.BatchNorm2d(256)
    self.pool = nn.MaxPool2d(kernel_size=2)

    # Dynamically calculating flattened size
    dummy_input = torch.zeros(1, 3, 32, 32)  # CIFAR-10 input size
    out = self.pool(self.bn1(self.activation(self.conv1(dummy_input))))
    out = self.pool(self.bn2(self.activation(self.conv2(out))))
    out = self.pool(self.bn3(self.activation(self.conv3(out))))
    self.flattened_size = out.numel()

    # Fully connected layers with Dropout
    self.fc1 = nn.Linear(self.flattened_size, 256)
    self.dropout = nn.Dropout(0.5)  
    self.fc2 = nn.Linear(256, 10)  

  def forward(self, x):
    # Forward through convolutional layers with activation, batch norm and pooling
    x = self.pool(self.bn1(self.activation(self.conv1(x))))
    x = self.pool(self.bn2(self.activation(self.conv2(x))))
    x = self.pool(self.bn3(self.activation(self.conv3(x))))

    x = x.view(x.size(0), -1)
    x = self.activation(self.fc1(x))
    x = self.dropout(x)
    x = self.fc2(x)
    return x
  
def train_model(model, train_loader, validation_loader, num_epochs=20, learning_rate=0.0005, device='cpu', step_size=5, gamma=0.1):
  model.to(device)  
  # Loss function
  criterion = nn.CrossEntropyLoss()
  # Optimizer  
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)  
  # Learning rate scheduler
  scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)  

  for epoch in range(num_epochs):
    # Setting the model to training mode
    model.train()  
    running_loss = 0.0
        
    # Training loop
    for i, (inputs, labels) in enumerate(train_loader, 0):
      inputs, labels = inputs.to(device), labels.to(device) 
            
      optimizer.zero_grad() 

      # Forward pass
      outputs = model(inputs)  

      # Calculating the loss
      loss = criterion(outputs, labels)
      # Backpropagation
      loss.backward()
      # Updating the model's parameters  
      optimizer.step()  
            
      running_loss += loss.item()

    # Printing the loss after each epoch
    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')

    # Validating the model on Validation data
    validate_model(model, validation_loader)
        
    # Step the scheduler after each epoch to adjust learning rate
    scheduler.step()  

  # Printing the current learning rate
  print(f"Learning Rate after epoch {epoch + 1}: {scheduler.get_last_lr()[0]}")

def validate_model(model, validation_loader, device='cpu'):
  # Set the model to evaluation mode
  model.eval()  
  correct = 0
  total = 0
    
  # Disabling gradient calculation for validation
  with torch.no_grad():  
    for inputs, labels in validation_loader:
      inputs, labels = inputs.to(device), labels.to(device)  #

      # Forward pass    
      outputs = model(inputs)  
      # Getting the class with the highest probability
      _, predicted = torch.max(outputs.data, 1)  
            
      total += labels.size(0)
      correct += (predicted == labels).sum().item()

  accuracy = 100 * correct / total
  print(f'Accuracy on validation set: {accuracy:.2f}%')
  return total, accuracy

def test_model(model, test_loader):
  predictions = []
  # Setting model to evaluation mode
  model.eval()  
  with torch.no_grad():
    for data in test_loader:
      # Reshaping the input data
      inputs = data[0].unsqueeze(1).float()  
      outputs = model(inputs)
      _, predicted = torch.max(outputs, 1)
      predictions.extend(predicted.cpu().numpy())
  return predictions

def load_and_test(model, device, test_loader):
  predictions = []
  true_labels = []

  # Running the model on test data

  # Disablings gradient calculation for inference
  with torch.no_grad():  
    for inputs, labels in test_loader:

      inputs, labels = inputs.to(device), labels.to(device)
        
      # Getting the model's predictions in the test data
      outputs = model(inputs)
        
      # Getting the predicted class 
      predicted_classes = outputs.argmax(dim=1)
        
      # Storing the predictions and true labels for evaluation
      predictions.extend(predicted_classes.cpu().numpy())
      true_labels.extend(labels.cpu().numpy())

  # Evaluating the model predictions
  accuracy = accuracy_score(true_labels, predictions)
  precision = precision_score(true_labels, predictions, average='weighted', zero_division=1)  
  recall = recall_score(true_labels, predictions, average='weighted', zero_division=1)
  f1 = f1_score(true_labels, predictions, average='weighted', zero_division=1)

  print(f"Test Accuracy: {accuracy * 100:.2f}%")
  print(f"Precision: {precision:.2f}")
  print(f"Recall: {recall:.2f}")
  print(f"F1 Score: {f1:.2f}")

  # Generating and saving the Confusion Matrix
  cm = confusion_matrix(true_labels, predictions)

  # Plot the confusion matrix
  plt.figure(figsize=(10, 10))
  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
  plt.title('Confusion Matrix')
  plt.xlabel('Predicted Labels')
  plt.ylabel('True Labels')
  plt.savefig('CNN-ConfusionMatrix')
    

    