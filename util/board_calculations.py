import numpy as np
from loguru import logger
from config import BoardConfig
import cv2 as cv
from . import Util
from scipy.ndimage import rotate


class BoardCalculations:

    @staticmethod
    def find_board(image: np.ndarray, conf: BoardConfig) -> np.ndarray:
        cv.imwrite(f'{conf.export.output_str}_image_org.png', image)

        res_image = Util.remove_black_writing(image)

        # find the tilt of the board
        right_tilt_angle, left_tilt_angle = Util.find_tilt_angle(res_image, conf.line_profile_width,
                                                                 conf.gaussian_sigma,
                                                                 10)
        res_image = rotate(res_image, (right_tilt_angle + left_tilt_angle) / 2, reshape=False)

        # find the edges of the board
        left_edge, right_edge = Util.find_board_edges(res_image, conf.line_profile_width, conf.gaussian_sigma,
                                                      conf.approx_board)
        logger.info(f"{conf.source_image}: Left_edge: {left_edge}, Right_edge:{right_edge}")
        cv.imwrite(f'{conf.export.output_str}_full_rotated.png', res_image)
        logger.info(f"{conf.source_image}: Printing full rotated board")

        # find the top/bottom of the board
        board_width = right_edge - left_edge
        tile_width = board_width // 8
        top_line = Util.find_top_line(res_image, left_edge, right_edge, tile_width)

        logger.info(f"{conf.source_image}: top line: {top_line}. Bottom line: {top_line + board_width}")
        if top_line < 0 or top_line + board_width > res_image.shape[0]:
            raise IndexError("The board is out of range")
        board = res_image[top_line + 10:top_line + board_width - 10, left_edge + 10:right_edge - 10]

        return board
