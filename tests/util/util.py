import os

import numpy as np

from util import Util


def test_convert_all_pdfs_in_paths():
    images = Util.convert_all_pdfs_in_paths(path='../../Demos/pdfs/')
    assert len(images) > 0
    assert len(list(images.items())[0][1]) == 2


def test_find_best_tile_line():
    width = 700
    image = np.ones((width, width, 1), )

    tile_width = 60
    image = Util.create_board(image=image, tile_width=60)

    right_edge = width // 2 + 4 * tile_width
    left_edge = width // 2 - 4 * tile_width

    y_line = Util.find_best_tile_line(image, right_edge=right_edge, left_edge=left_edge, tile_width=tile_width)
    assert y_line == width // 2 + tile_width // 2


def test__find_top_line():
    width, height = 700, 700
    image = np.ones((width, height, 1), )

    tile_width = 60
    image = Util.create_board(image=image, tile_width=60)
    right_edge = width // 2 + 4 * tile_width
    left_edge = width // 2 - 4 * tile_width
    tile_center_line = width // 2 + tile_width // 2
    top_line = Util._find_top_line(image, right_edge=right_edge, left_edge=left_edge, tile_width=tile_width,
                                   tile_center_line=tile_center_line)
    assert top_line == height // 2 - 4 * tile_width


def test_split_black_white_pieces():
    for folder in os.scandir(r"D:\Chess_problem_converter\chess_problem_converter\out"):
        if not folder.is_dir():
            continue
        print(folder.path)
        Util.split_black_white_pieces(folder.path)

