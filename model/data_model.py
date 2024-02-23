from loguru import logger
import os
import pickle

import cv2 as cv
from scipy.ndimage import rotate
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
class PiecesDataset(Dataset):
    def __init__(self, images, classifications):

        self.images = images
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
        self.conv1 = nn.Conv2d(3, 6, 5) # Assuming RGB images
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10) # Assuming 10 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


names = [
    "Nearest Neighbors",
    "Random Forest",
    "Neural Net",
]
classifiers = [
    KNeighborsClassifier(3),
    RandomForestClassifier(
        max_depth=5, n_estimators=10, max_features=1, random_state=42
    ),
    MLPClassifier(alpha=1, max_iter=1000, random_state=42),
]


class DataModel:

    @staticmethod
    def train_from_images():
        tile_size = 32
        logger.debug("starting training:")
        training_lib = "trainingset"
        piece_lib_dict = DataModel.split_paths_to_dict(paths=os.walk(training_lib))

        all_pieces = {}
        for pieces, lib in piece_lib_dict.items():
            all_pieces[pieces] = [cv.cvtColor(cv.imread(f.path), cv.COLOR_BGR2GRAY) for f in os.scandir(lib) if
                                  f.is_file()]
            all_pieces[pieces] = DataModel.grow_dataset(all_pieces[pieces])
            all_pieces[pieces] = [cv.resize(image, (tile_size, tile_size), interpolation=cv.INTER_LINEAR) for image in
                                  all_pieces[pieces]]

        with open(f'piece_lib_dict.pkl', 'wb') as f:  # open a text file
            pickle.dump({type_nr: piece_type for type_nr, piece_type in enumerate(piece_lib_dict.keys())}, f)

        data, classification = DataModel.dict_to_datasets(piece_lib_dict=piece_lib_dict, all_pieces=all_pieces)

        x_train, x_test, y_train, y_test = train_test_split(data, classification, test_size=0.5, shuffle=True)

        DataModel.train_models(x_train, y_train)

        for name, clf in zip(names, classifiers):
            score = clf.score(x_test, y_test)
            print(f"{name} : {score}")





    def init_torch_model(self):
        # Instantiate your model, loss function, and optimizer
        self.model = SimpleCNN()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)


    def train_torch_model(self):
        tile_size = 32
        logger.debug("starting training:")
        training_lib = "trainingset"
        piece_lib_dict = DataModel.split_paths_to_dict(paths=os.walk(training_lib))

        all_pieces = {}
        for pieces, lib in piece_lib_dict.items():
            all_pieces[pieces] = [cv.cvtColor(cv.imread(f.path), cv.COLOR_BGR2GRAY) for f in os.scandir(lib) if
                                  f.is_file()]
            all_pieces[pieces] = DataModel.grow_dataset(all_pieces[pieces])
            all_pieces[pieces] = [cv.resize(image, (tile_size, tile_size), interpolation=cv.INTER_LINEAR) for image in
                                  all_pieces[pieces]]

        with open(f'piece_lib_dict.pkl', 'wb') as f:  # open a text file
            pickle.dump({type_nr: piece_type for type_nr, piece_type in enumerate(piece_lib_dict.keys())}, f)

        data, classification = DataModel.dict_to_datasets(piece_lib_dict=piece_lib_dict, all_pieces=all_pieces)
        images_tensor = torch.tensor(data, dtype=torch.float32)
        labels_tensor = torch.tensor(classification, dtype=torch.long)
        my_dataset = PiecesDataset(images_tensor, labels_tensor)

        # Create a DataLoader
        data_loader = DataLoader(my_dataset, batch_size=4, shuffle=True, num_workers=2)

        # Assuming you have a DataLoader for your training data called trainloader
        for epoch in range(10):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, inputs, labels in enumerate(data_loader):
                # inputs, labels = data

                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1}]: loss: {running_loss / 2000}')
                    running_loss = 0.0

        print('Finished Training')

    @staticmethod
    def grow_dataset(images: list[np.ndarray]) -> list[np.ndarray]:

        kernels = DataModel.create_shift_kernels(5)
        images = DataModel.grow_though_rotate(images=images, angle=4)
        return DataModel.grow_though_kernels(images=images, kernels=kernels)

    @staticmethod
    def create_shift_kernels(size: int) -> np.ndarray:
        """returns the kernels used to shift the image one pixel in all directions (incl no directions)"""
        kernels = np.array([np.zeros((size, size), np.int8) for i in range(size * size)])
        for x in range(size):
            for y in range(size):
                kernels[x * size + y][x][y] = 1
        return kernels

    @staticmethod
    def grow_though_rotate(images: list[np.ndarray], angle: int) -> list[np.ndarray]:
        return [rotate(img, deg, reshape=False) for img in images for deg in range(-angle, angle + 1)]

    @staticmethod
    def grow_though_kernels(images: list[np.ndarray], kernels: np.ndarray) -> list[np.ndarray]:
        return [cv.filter2D(src=img, ddepth=-1, kernel=kernel) for img in images for kernel in kernels]

    @staticmethod
    def train_models(x, y) -> None:
        for name, clf in zip(names, classifiers):
            clf = make_pipeline(StandardScaler(), clf)
            clf.fit(x, y)
            with open(f'models/{name}.pkl', 'wb') as f:  # open a text file
                pickle.dump(clf, f)  # serialize the list

    @staticmethod
    def dict_to_datasets(piece_lib_dict: dict[int, str], all_pieces: dict[str, list[np.ndarray]]) -> (
            list[np.ndarray], list[int]):
        data = []
        classification = []
        for type_nr, piece_type in enumerate(piece_lib_dict.keys()):
            for piece in all_pieces[piece_type]:
                data.append(piece.reshape(-1))
                classification.append(type_nr)
        return data, classification

    @staticmethod
    def split_paths_to_dict(paths: list[str]):
        return {x[0].split('\\')[-1] + "_" + x[0].split('\\')[-2]: x[0] for x in paths if len(x[0].split("\\")) > 2}
