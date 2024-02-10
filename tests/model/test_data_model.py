import numpy as np
from chess_problem_converter.model.data_model import DataModel


def test_grow_data() -> None:
    images = np.ones((64, 64, 1))

    assert len(DataModel.grow_dataset(images)) > 1


def test_create_shift_kernels()->None:
    length, x, y = DataModel.create_shift_kernels(5).shape
    assert length == 25
    assert x == y == 5
    assert np.all(sum(DataModel.create_shift_kernels(5)) == np.ones((5, 5)))


def test_grow_though_rotate()->None:
    images = [np.ones((64, 64)), np.zeros((64, 64))]
    rotated_images = DataModel.grow_though_rotate(images=images, angle=5)
    assert len(rotated_images) == len(images) * (2 * 5 + 1)

def test_split_paths_to_dict()->None:
    list_of_paths = [('trainingset\\Knight\\Black', [], []),('trainingset\\Knight\\White', [], ["1.png"])]
    spilt_dict = DataModel.split_paths_to_dict(list_of_paths)
    expected = {"Black_Knight":"trainingset\\Knight\\Black", "White_Knight":"trainingset\\Knight\\White"}
    assert spilt_dict==expected