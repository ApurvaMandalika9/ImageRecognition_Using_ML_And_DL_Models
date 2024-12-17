from torchvision import datasets, models
import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

class ResNet(nn.Module):
    def __init__(self, device='cpu', learning_rate=0.001, num_filters=64, kernel_size=3, optimizer='adam', activation='relu'):
        super(ResNet, self).__init__()

        self.model_path = "./models/resnet.pth"
        self.model = models.resnet18(pretrained=True)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=num_filters, 
                               kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)
        self.device = device
        self.model = self.model.to(device)

        self.criterion = nn.CrossEntropyLoss()

        if optimizer == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        else:
            self.optimizer = optim.Adadelta(self.model.parameters(), lr=learning_rate)

        # Set the activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()

        self.hyperparameters = {
            "activation": activation,
            "learning_rate": learning_rate,
            "num_filters": num_filters,
            "optimizer": optimizer,
            "kernel_size": kernel_size
        }

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.activation(x)  # Activation function
        x = self.model.bn1(x)   # Batch normalization
        x = self.model.relu(x)  # ReLU after batch norm
        x = self.model.maxpool(x)  # Maxpooling
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.fc(x)

        return x

    def train_model(self, train_loader, validation_loader, num_epochs=10):
        print("\nTraining ResNet:")

        # Training the model
        for epoch in range(num_epochs):
            self.model.train()  # Set model to training mode
            running_loss = 0.0
            correct = 0
            total = 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()  # Zero the parameter gradients

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()  # Backpropagation

                self.optimizer.step()  # Update model parameters

                running_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print(f"Epoch {epoch+1}/{num_epochs}\nLoss: {running_loss/len(train_loader)}, Accuracy: {100 * correct / total}%")
            _, accuracy = self.validate_model(validation_loader)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'hyperparameters': self.hyperparameters,
            'accuracy': accuracy
        }, self.model_path)

    def validate_model(self, validation_loader):
        self.model.eval()  # Set model to evaluation mode
        correct = 0
        total = 0
        y_pred = []
        with torch.no_grad():
            for inputs, labels in validation_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                y_pred.extend(predicted)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Validation Accuracy: {accuracy}%")
        return y_pred, accuracy

    def test_model(self, test_loader):
        print("\nTesting ResNet:")
        # self.model.eval()
        # correct = 0
        # total = 0
        # y_pred = []
        # with torch.no_grad():
        #     for inputs, labels in test_loader:
        #         inputs, labels = inputs.to(self.device), labels.to(self.device)
        #         outputs = self.model(inputs)
        #         _, predicted = torch.max(outputs, 1)
        #         y_pred.extend(predicted)
        #         total += labels.size(0)
        #         correct += (predicted == labels).sum().item()

        # accuracy = 100 * correct / total
        # print(f"Test Accuracy: {accuracy}%")
        # return y_pred, accuracy
        predictions = []
        true_labels = []
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        # Run the model on test data
        with torch.no_grad():  # Disable gradient calculation for inference
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Get the model's predictions
                outputs = self.model(inputs)
                
                # Get the predicted class
                predicted_classes = outputs.argmax(dim=1)
                
                # Store predictions and true labels for evaluation
                predictions.extend(predicted_classes.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        # Evaluate the model
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, average='weighted', zero_division=1)
        recall = recall_score(true_labels, predictions, average='weighted', zero_division=1)
        f1 = f1_score(true_labels, predictions, average='weighted', zero_division=1)
        print(f"Test Accuracy: {accuracy * 100:.2f}%")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1 Score: {f1:.2f}")

        # Confusion Matrix
        cm = confusion_matrix(true_labels, predictions)

        # Plot the confusion matrix
        plt.figure(figsize=(10, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.savefig('ResNet-ConfusionMatrix')

        return predictions
    


        