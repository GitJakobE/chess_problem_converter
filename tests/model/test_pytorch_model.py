import os

import numpy as np
import torch

from model import TouchDataModel, PiecesDataset, SimpleCNN


def test_pieces_dataset():
    images = torch.rand(10, 3, 32, 32)
    classifications = torch.randint(0, 15, (10,))  # 15 classes with 2 alt bishop and an empty
    dataset = PiecesDataset(images, classifications)

    assert len(dataset) == 10
    image, classification = dataset[0]
    assert image.shape == torch.Size([3, 32, 32])
    assert classification in range(15)


def test_model_forward_pass():
    model = SimpleCNN()
    dummy_input = torch.rand(1, 3, 32, 32)  # Single RGB image
    output = model(dummy_input)
    assert output.shape == torch.Size([1, 15])  # 15 classes with 2 alt bishop and an empty


def test_train_pytorch() -> None:
    dm = TouchDataModel()
    dm.init_torch_model()
    dm.train_torch_model(training_lib="../../trainingset")
    dm.save_model()


def test_model_evaluation():
    touch_data_model = TouchDataModel()
    touch_data_model.init_torch_model()
    touch_data_model.load_model("../../model_weights.pth")

    dummy_image = np.random.randint(255, size=(32, 32, 3), dtype=np.uint8)
    probabilities = touch_data_model.evaluate(dummy_image)
    assert len(probabilities) == 15
    assert sum(probabilities) - 1 < 1e-5  # Sum of probabilities should be close to 1


def test_load_save_data_model() -> None:
    dm = TouchDataModel()
    dm.init_torch_model()
    filename = "test.pth"
    dm.save_model(filename)
    assert os.path.exists(filename)
    dm.load_model(filename)

    os.unlink(filename)
