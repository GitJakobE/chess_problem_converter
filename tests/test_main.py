from util import Util
import cv2 as cv

from scipy.ndimage import rotate

demos = [("Toft-00-0000.png", 11, 0, 1,)]


def test_left_right_demos():
    for demo in demos:
        image = cv.imread(f"../Demos/{demo[0]}")
        image = cv.resize(image, (612, 792), interpolation=cv.INTER_LINEAR)
        res_image = Util.remove_black_writing(image)
        right_tilt_angle, left_tilt_angle = Util.find_tilt_angle(res_image, 300, 5, 10)
        assert abs(right_tilt_angle) <= demo[3] and abs(left_tilt_angle) <= demo[3] and abs(right_tilt_angle) >= demo[
            2] and abs(left_tilt_angle) >= demo[2]

        res_image = rotate(res_image, (right_tilt_angle + left_tilt_angle) / 2, reshape=False)

        # find the edges of the board
        left_edge, right_edge = Util.find_board_edges(res_image, 300, 5,510)

        # find the top/bottom of the board
        board_width = right_edge - left_edge
        tile_width = board_width // 8
        top_line = Util.find_top_line(res_image, left_edge, right_edge, tile_width)


    Util.find_tilt_angle()
