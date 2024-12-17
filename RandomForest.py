import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def rf_train_model(x_train, y_train, x_valid, y_valid):
    print("\nTraining Random Forest:")

    # Hyperparameter tuning parameters grid
    param_grid = {
        'criterion': ['gini', 'log_loss'],
        'max_depth': [5, 10, 100, 10000, None],
        'min_samples_leaf': [1, 5, 10, 50],
        'max_features': ["sqrt", "log2"]
    }

    model = RandomForestClassifier(
        criterion=param_grid['criterion'][0],
        max_depth=param_grid['max_depth'][4],
        max_features=param_grid['max_features'][0],
        min_samples_leaf=param_grid['min_samples_leaf'][2],
        random_state=42
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_valid)
    accuracy_test = accuracy_score(y_valid, y_pred)
    print(f"Accuracy on Validation Data: {accuracy_test * 100:.2f}%")

    model_filename = 'models/random_forest_model.pkl'
    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)
    print("Model Saved")

    return model

def rf_test_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    accuracy_test = accuracy_score(y_test, y_pred)
    print(f"Accuracy on Test Data: {accuracy_test * 100:.2f}%")
    return y_pred, accuracy_test

def rf_test_best_model(x_train, y_train, x_valid, y_valid, x_test, y_test):

    print("\nTesting Best Random Forest Model:")

    model_filename = 'random_forest_best.pkl'
    with open(model_filename, 'rb') as file:
        model = pickle.load(file)
    print("Model Loaded")
    print(model)
    y_pred = model.predict(x_test)
    accuracy_test = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=1) 
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)
    print(f"Test Accuracy: {accuracy_test * 100:.2f}%")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot the confusion matrix
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig('RandomForest-ConfusionMatrix')




