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


class data_model:


    @staticmethod
    def train_from_images():
        logger.debug("starting training:")
        training_lib = "trainingset"
        piece_lib_dict = {x[0].split('\\')[-1] + "_" + x[0].split('\\')[-2]: x[0] for x in os.walk(training_lib) if
                          len(x[0].split("\\")) > 2}

        kernels = np.array([np.zeros((5, 5), np.int8) for i in range(25)])
        for x in range(5):
            for y in range(5):
                kernels[x * 5 + y][x][y] = 1

        all_pieces = {}
        for pieces, lib in piece_lib_dict.items():
            all_pieces[pieces] = [cv.cvtColor(cv.imread(f.path), cv.COLOR_BGR2GRAY) for f in os.scandir(lib) if
                                  f.is_file()]
            # grow the dataset:

            # rotate +/- 4 degrees
            grown_data = []
            for deg in range(-4, 5):
                for img in all_pieces[pieces]:
                    grown_data.append(rotate(img, deg, reshape=False))
            all_pieces[pieces] = grown_data
            grown_data = []
            # move +/-4 pixels
            for img in all_pieces[pieces]:
                for kernel in kernels:
                    grown_data.append(cv.filter2D(src=img, ddepth=-1, kernel=kernel))

            all_pieces[pieces] = [cv.resize(image, (32, 32), interpolation=cv.INTER_LINEAR) for image in grown_data]
            # think about making it a binary image
            # th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
        print(all_pieces)
        data = []
        classification = []
        n = 0

        with open(f'piece_lib_dict.pkl', 'wb') as f:  # open a text file
            pickle.dump({type_nr: piece_type for type_nr, piece_type in enumerate(piece_lib_dict.keys())}, f)
        for type_nr, piece_type in enumerate(piece_lib_dict.keys()):
            for piece in all_pieces[piece_type]:
                data.append(piece.reshape(-1))
                classification.append(type_nr)
                n += 1
                cv.imwrite(f'trainingset/{piece_type}_{type_nr}/{n}.png', piece)

        # serialize the list
        print(len(data))

        X_train, X_test, y_train, y_test = train_test_split(data, classification, test_size=0.5, shuffle=True)
        # clf = svm.SVC(gamma=0.001)
        names = [
            "Nearest Neighbors",
            "Random Forest",
            "Neural Net",
        ]
        classifiers = [
            KNeighborsClassifier(3),
            KNeighborsClassifier(3),
            RandomForestClassifier(
                max_depth=5, n_estimators=10, max_features=1, random_state=42
            ),
            MLPClassifier(alpha=1, max_iter=1000, random_state=42),
        ]
        for name, clf in zip(names, classifiers):
            clf = make_pipeline(StandardScaler(), clf)
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            print(f"{name} : {score}")
            with open(f'models/{name}.pkl', 'wb') as f:  # open a text file
                pickle.dump(clf, f)  # serialize the list
