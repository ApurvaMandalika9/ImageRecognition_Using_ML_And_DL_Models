import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

def load_from_pickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_data(data_dir):
    # train data
    file_prefix = f"{data_dir}/data_batch_"
    for i in range(1,6):
        data = load_from_pickle(file_prefix+str(i))
        if i==1:
            x_train = data[b"data"]
            y_train = data[b"labels"]
        else:
            x_train = np.concatenate((x_train,data[b"data"]),axis=0)
            y_train.extend(data[b'labels'])
    y_train = np.array(y_train)

    # test data
    file = f"{data_dir}/test_batch"
    data = load_from_pickle(file)
    x_test = data[b"data"]
    y_test = np.array(data[b"labels"])
    print(f"x_train: {x_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"x_test: {x_test.shape}")
    print(f"y_test: {y_test.shape}")

    return x_train, y_train, x_test, y_test

def train_valid_split(x_train, y_train, train_ratio=0.8):

    split_index = int(np.multiply(x_train.shape[0], train_ratio))

    x_train_new = x_train[:split_index]
    y_train_new = y_train[:split_index]
    x_valid = x_train[split_index:]
    y_valid = y_train[split_index:]

    return x_train_new, y_train_new, x_valid, y_valid

class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Convert the raw data from (N, 3072) into a 3x32x32 image
        image = self.data[idx].reshape(3, 32, 32).transpose((1, 2, 0))  # Reshape and change channel order to (C, H, W)
        image = Image.fromarray(image)  # Convert the numpy array into a PIL Image
        
        # Apply the transformation (e.g., normalization)
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        return image, label