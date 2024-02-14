import numpy as np

from util import Util


def test_convert_all_pdfs_in_paths():
    images = Util.convert_all_pdfs_in_paths(path='../../Demos/pdfs/')
    assert len(images) > 0
    assert len(list(images.items())[0][1]) == 2


def create_board(image: np.ndarray, tile_width: int) -> np.ndarray:
    height, width, _ = image.shape
    if width // 2 - tile_width * 4 < 0:
        print("out of range")
    start_x = width // 2 - tile_width * 4
    start_y = height // 2 - tile_width * 4
    for tile_x in range(8):
        for tile_y in range(8):
            tile_color = 128
            if (tile_x % 2 == 0 and tile_y % 2 != 0) or (tile_x % 2 != 0 and tile_y % 2 == 0):
                tile_color = 255

            for x in range(tile_width):
                for y in range(tile_width):
                    image[start_x + x + tile_x * tile_width][start_y + y + tile_y * tile_width] = tile_color
    return image


def test_find_best_tile_line():
    width = 700
    image = np.ones((width, width, 1), )

    tile_width = 60
    image = create_board(image=image, tile_width=60)

    right_edge = width // 2 + 4 * tile_width
    left_edge = width // 2 - 4 * tile_width

    y_line = Util.find_best_tile_line(image, right_edge=right_edge, left_edge=left_edge, tile_width=tile_width)
    assert y_line == width // 2 + tile_width // 2


def test__find_top_line():
    width, height = 700, 700
    image = np.ones((width, height, 1), )

    tile_width = 60
    image = create_board(image=image, tile_width=60)
    right_edge = width // 2 + 4 * tile_width
    left_edge = width // 2 - 4 * tile_width
    tile_center_line = width // 2 + tile_width // 2
    top_line = Util._find_top_line(image, right_edge=right_edge, left_edge=left_edge, tile_width=tile_width,
                                   tile_center_line=tile_center_line)
    assert top_line == height // 2 - 4 * tile_width
