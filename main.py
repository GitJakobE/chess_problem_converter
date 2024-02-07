import cv2 as cv
import sys

from util import Util
from scipy.ndimage import rotate
from loguru import logger
from config import BoardConfig
import os
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

source_image = "data\Toft-00-0174.png" if len(sys.argv) < 2 else sys.argv[1]


# Main function
def main():
    conf = BoardConfig()
    conf.read_predict_dict()
    logger.add("out/log.log")

    source_images = [filename.path for filename in os.scandir("data/") if filename.path.endswith(".png")]

    for source_image in source_images:
        conf.set_export(source_image)
        image = cv.imread(source_image)

        logger.info(f"{source_image}: Printing board")
        cv.imwrite(f'{conf.export.output_str}_image_org.png', image)

        image = Util.remove_black_writing(image)
        # find the tilt of the board
        right_tilt_angle, left_tilt_angle = Util.find_tilt_angle(image, conf.line_profile_width, conf.gaussian_sigma, 10)
        image = rotate(image, (right_tilt_angle + left_tilt_angle) / 2, reshape=False)

        # find the edges of the board
        left_edge, right_edge = Util.find_board_edges(image, conf.line_profile_width, conf.gaussian_sigma,
                                                      conf.approx_board)
        logger.info(f"{source_image}: Left_edge: {left_edge}, Right_edge:{right_edge}")
        cv.imwrite(f'{conf.export.output_str}_full_rotated.png', image)
        logger.info(f"{source_image}: Printing full rotated board")

        # find the top/bottom of the board
        board_width = right_edge - left_edge
        tile_width = board_width // 8
        top_line = Util.find_top_line(image, left_edge, right_edge, tile_width)

        board = image[top_line + 10:top_line + board_width - 10, left_edge + 10:right_edge - 10]
        logger.info(f"{source_image}: top line: {top_line}. Bottom line: {top_line + board_width}")
        if top_line<0 or top_line + board_width>image.shape[0]:
            logger.error(f"{source_image}: board out of scope:")
            continue
        a = os.scandir("models/")
        # models = [model for model in a if model.name.endswith(".pkl")]
        # for model in models:
        logger.info(f"printing board and pieces for {source_image}")
        cv.imwrite(f'{conf.export.output_str}_board.png', board)

        # with open("models/Nearest Neighbors.pkl", 'rb') as f:  # open a text file
        #     clf = pickle.load(f)
        # Util.print_board(board=board, conf=conf, classifier=clf)

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
        all_pieces[pieces] = [cv.cvtColor(cv.imread(f.path), cv.COLOR_BGR2GRAY) for f in os.scandir(lib) if f.is_file()]
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
        pickle.dump({type_nr:piece_type  for type_nr, piece_type in enumerate(piece_lib_dict.keys())}, f)
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


# Run the program
# train_from_images()
main()
