import pytest

import cv2 as cv

from util import BoardCalculations, Board, Util, constants
from config import BoardConfig

demos = [
    # Board(name="Toft-00-0000.png", l_edge=46, r_edge=535, r_tilt_angle=0, l_tilt_angle=0, top_line=114,
    #       nr_of_pieces=11),
    # Board(name="Toft-00-0001.png", l_edge=68, r_edge=556, r_tilt_angle=0, l_tilt_angle=0, top_line=117,
    #       nr_of_pieces=16),
    # Board(name="Toft-00-0002.png", l_edge=31, r_edge=575, r_tilt_angle=0, l_tilt_angle=0, top_line=137,
    #       nr_of_pieces=19),
    # Board(name="Toft-00-0003.png", l_edge=49, r_edge=537, r_tilt_angle=0, l_tilt_angle=0, top_line=112,
    #       nr_of_pieces=17),
    # Board(name="Toft-00-0004.png", l_edge=57, r_edge=539, r_tilt_angle=0, l_tilt_angle=0, top_line=107,
    #       nr_of_pieces=16),
    # Board(name="Toft-00-0005.png", l_edge=51, r_edge=572, r_tilt_angle=0, l_tilt_angle=0, top_line=121,
    #       nr_of_pieces=22),
    # Board(name="Toft-00-0008.png", l_edge=48, r_edge=568, r_tilt_angle=0, l_tilt_angle=0, top_line=114,
    #       nr_of_pieces=14),
    # Board(name="Toft-00-0042.png", l_edge=48, r_edge=556, r_tilt_angle=0, l_tilt_angle=0, top_line=114,
    #       nr_of_pieces=12),
    # Board(name="Toft-00-0270.png", l_edge=62, r_edge=560, r_tilt_angle=0, l_tilt_angle=0, top_line=126,
    #       nr_of_pieces=18),
    # Board(name="Toft-00-0338.png", l_edge=60, r_edge=556, r_tilt_angle=0, l_tilt_angle=0, top_line=121,
    #       nr_of_pieces=13),
    Board(name="Toft-00-0130.png", l_edge=88, r_edge=488, r_tilt_angle=0, l_tilt_angle=0, top_line=124,
          nr_of_pieces=23),
    Board(name="Toft-00-0066.png", l_edge=88, r_edge=488, r_tilt_angle=0, l_tilt_angle=0, top_line=124,
          nr_of_pieces=23),

]


@pytest.mark.parametrize("demo", demos)
def test_demos(demo: Board):
    image = cv.imread(f"../Demos/{demo.name}")
    image = cv.resize(image, constants.standard_image_dim, interpolation=cv.INTER_LINEAR)
    conf = BoardConfig()
    conf.set_export(demo.name)
    conf.set_source_image(demo.name)
    conf.init_torch_model('../models/model_weights.pth')
    conf.read_predict_dict('../models/piece_lib_dict.pkl')

    board = BoardCalculations.find_board(image, conf)
    Util.find_pieces(board, conf)
    assert abs(board.r_tilt_angle - demo.r_tilt_angle) <= 1.0 and abs(
        board.l_tilt_angle - demo.l_tilt_angle) <= 1.0
    assert abs(board.l_edge - demo.l_edge) < 5 and abs(board.r_edge - demo.r_edge) < 5
    assert abs(board.top_line - demo.top_line) < 5
    assert len(board.pieces) == demo.nr_of_pieces
