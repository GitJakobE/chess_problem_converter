import os
import pickle
import cv2 as cv

from model.pytorch_model import TouchDataModel


def test_train_pytorch() -> None:
    dm = TouchDataModel()
    dm.init_torch_model()
    dm.train_torch_model(training_lib="../../trainingset")
    dm.save_model()


def test_load_save_data_model() -> None:
    dm = TouchDataModel()
    dm.init_torch_model()
    filename = "test.pth"
    dm.save_model(filename)
    dm.load_model(filename)
    assert os.path.exists(filename)
    os.unlink(filename)


def test_evaluate() -> None:
    dm = TouchDataModel()
    dm.init_torch_model()
    dm.load_model()
    with open("../../piece_lib_dict.pkl", 'rb') as f:  # open a text file
        predict_dict = pickle.load(f)


    image = cv.imread("..\\..\\out\\Toft-00-0000_b8.png")
    lib = predict_dict[int(dm.evaluate(image=image))]
    print(lib)