from util import Util, BoardCalculations, Board
from config import BoardConfig
import cv2 as cv
from scipy.ndimage import rotate

# demos = [(11, 0, 1, 40, 47, 535,541)]
demos = [
    Board(name="Toft-00-0000.png", left_edge=43, right_edge=535, right_tilt_angle=0, left_tilt_angle=0, top_line=105),
    Board(name="Toft-00-0001.png", left_edge=60, right_edge=560, right_tilt_angle=0, left_tilt_angle=0, top_line=111),
    Board(name="Toft-00-0002.png", left_edge=23, right_edge=582, right_tilt_angle=0, left_tilt_angle=0, top_line=123),
    Board(name="Toft-00-0003.png", left_edge=42, right_edge=543, right_tilt_angle=0, left_tilt_angle=0, top_line=107),
    Board(name="Toft-00-0004.png", left_edge=52, right_edge=539, right_tilt_angle=0, left_tilt_angle=0, top_line=103),
    Board(name="Toft-00-0005.png", left_edge=38, right_edge=582, right_tilt_angle=0, left_tilt_angle=0, top_line=107),
    Board(name="Toft-00-0008.png", left_edge=38, right_edge=578, right_tilt_angle=0, left_tilt_angle=0, top_line=102),
]


def test_left_right_demos():
    for demo in demos[:5]:
        image = cv.imread(f"../Demos/{demo.name}")
        image = cv.resize(image, (612, 792), interpolation=cv.INTER_LINEAR)
        conf = BoardConfig()
        conf.set_export(demo.name)
        conf.set_source_image(demo.name)

        board = BoardCalculations.find_board(image, conf)
        assert abs(board.right_tilt_angle - demo.right_tilt_angle) <= 1.0 and abs(
            board.left_tilt_angle - demo.left_tilt_angle) <= 1.0
        assert abs(board.left_edge - demo.left_edge) < 5 and abs(board.right_edge - demo.right_edge) < 5
        assert abs(board.top_line - demo.top_line) < 8
