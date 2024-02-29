from typing import List
from PIL import Image
import numpy as np
import os
import pickle

import cv2 as cv
from loguru import logger
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from .data_model import DataModel


class PiecesDataset(Dataset):
    def __init__(self, images, classifications):
        self.images = images  # .unsqueeze(1)
        self.classifications = classifications

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        classification = self.classifications[idx]
        return image, classification


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  # Assuming RGB images
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 15)  # Assuming 12 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class TouchDataModel:

    def init_torch_model(self):
        self.model = SimpleCNN()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        self.tile_size = 32
        self.train_transform = transforms.Compose([
            transforms.Resize((self.tile_size, self.tile_size)),
            # transforms.Grayscale(),
            transforms.RandomRotation(3),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Random translation
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        self.transform = transforms.Compose([
            transforms.Resize((self.tile_size, self.tile_size)),
            # transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

    def train_torch_model(self, training_lib):

        logger.debug("starting training:")
        piece_lib_dict = DataModel.split_paths_to_dict(paths=os.walk(training_lib))
        all_pieces = {}
        for pieces, lib in piece_lib_dict.items():
            all_pieces[pieces] = [self.train_transform(Image.open(f.path)) for f in os.scandir(lib) if
                                  f.is_file()]
            # all_pieces[pieces] = DataModel.grow_dataset(images=all_pieces[pieces], kernelshift=3, angles=2)
            # all_pieces[pieces] = [Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB)) for img in all_pieces[pieces]]


        with open(f'piece_lib_dict.pkl', 'wb') as f:  # open a text file
            pickle.dump({type_nr: piece_type for type_nr, piece_type in enumerate(piece_lib_dict.keys())}, f)

        data, classification = DataModel.dict_to_datasets(piece_lib_dict=piece_lib_dict, all_pieces=all_pieces,
                                                          flatten=False)
        # images_tensor = torch.tensor(data, dtype=torch.float32)
        labels_tensor = torch.tensor(classification, dtype=torch.long)
        my_dataset = PiecesDataset(data, labels_tensor)

        data_loader = DataLoader(my_dataset, batch_size=16, shuffle=True, num_workers=2)

        for epoch in range(40):
            running_loss = 0.0
            for i, data in enumerate(data_loader):
                inputs, labels = data

                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if i % 100 == 99:  # print every 100 mini-batches
                    print(f'[{epoch + 1}, {i + 1}]: loss: {running_loss / 100}')
                    running_loss = 0.0

        print('Finished Training')
        torch.save(self.model.state_dict(), 'model_weights.pth')

    def evaluate(self, image: np.ndarray) -> List[float]:
        self.model.eval()
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        image_tensor = self.transform(image).unsqueeze(0)
        with torch.no_grad():  # No need to track gradients
            output = self.model(image_tensor)
        probabilities = torch.softmax(output, dim=1)

        return probabilities.squeeze().tolist()

    def save_model(self, name: str = 'model_weights.pth'):
        torch.save(self.model.state_dict(), name)

    def load_model(self, name: str = 'model_weights.pth'):
        self.model.load_state_dict(torch.load(name))
