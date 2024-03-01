import pytest

import cv2 as cv

from util import BoardCalculations, Board, Util
from config import BoardConfig

demos = [
    Board(name="Toft-00-0000.png", left_edge=46, right_edge=535, right_tilt_angle=0, left_tilt_angle=0, top_line=114,
          nr_of_pieces=11),
    Board(name="Toft-00-0001.png", left_edge=68, right_edge=556, right_tilt_angle=0, left_tilt_angle=0, top_line=117,
          nr_of_pieces=16),
    Board(name="Toft-00-0002.png", left_edge=31, right_edge=575, right_tilt_angle=0, left_tilt_angle=0, top_line=137,
          nr_of_pieces=19),
    Board(name="Toft-00-0003.png", left_edge=49, right_edge=537, right_tilt_angle=0, left_tilt_angle=0, top_line=112,
          nr_of_pieces=17),
    Board(name="Toft-00-0004.png", left_edge=57, right_edge=539, right_tilt_angle=0, left_tilt_angle=0, top_line=107,
          nr_of_pieces=16),
    Board(name="Toft-00-0005.png", left_edge=51, right_edge=572, right_tilt_angle=0, left_tilt_angle=0, top_line=121,
          nr_of_pieces=22),
    Board(name="Toft-00-0008.png", left_edge=48, right_edge=568, right_tilt_angle=0, left_tilt_angle=0, top_line=114,
          nr_of_pieces=14),
    Board(name="Toft-00-0042.png", left_edge=48, right_edge=556, right_tilt_angle=0, left_tilt_angle=0, top_line=114,
          nr_of_pieces=12),
    Board(name="Toft-00-0270.png", left_edge=62, right_edge=560, right_tilt_angle=0, left_tilt_angle=0, top_line=126,
          nr_of_pieces=18),
    Board(name="Toft-00-0338.png", left_edge=60, right_edge=556, right_tilt_angle=0, left_tilt_angle=0, top_line=121,
          nr_of_pieces=13),
]


@pytest.mark.parametrize("demo", demos)
def test_demos(demo: Board):
    image = cv.imread(f"../Demos/{demo.name}")
    image = cv.resize(image, (612, 792), interpolation=cv.INTER_LINEAR)
    conf = BoardConfig()
    conf.set_export(demo.name)
    conf.set_source_image(demo.name)
    conf.init_torch_model('../models/model_weights.pth')
    conf.read_predict_dict('../models/piece_lib_dict.pkl')

    board = BoardCalculations.find_board(image, conf)
    Util.find_pieces(board, conf)
    assert abs(board.right_tilt_angle - demo.right_tilt_angle) <= 1.0 and abs(
        board.left_tilt_angle - demo.left_tilt_angle) <= 1.0
    assert abs(board.left_edge - demo.left_edge) < 5 and abs(board.right_edge - demo.right_edge) < 5
    assert abs(board.top_line - demo.top_line) < 5
    assert len(board.pieces) == demo.nr_of_pieces
