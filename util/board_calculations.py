import numpy as np
from loguru import logger
from config import BoardConfig
import cv2 as cv
from . import Util
from scipy.ndimage import rotate
from .Board import Board


class BoardCalculations:

    @staticmethod
    def find_board(image: np.ndarray, conf: BoardConfig) -> Board:
        """finds the board and returns it with the cal. images"""
        cv.imwrite(f'{conf.export.output_str}_image_org.png', image)

        board = Board(org_image=image, board_width=conf.approx_board_width)

        res_image = Util.remove_black_writing(image)

        # find the tilt of the board
        right_angle, left_angle = Util.find_tilt_angle(res_image, conf.line_profile_width, conf.gaussian_sigma, 10)
        board.right_tilt_angle, board.left_tilt_angle = BoardCalculations.set_to_min(right_angle, left_angle, 1.0)

        board.rotated_image = rotate(res_image, (board.right_tilt_angle + board.left_tilt_angle) / 2, reshape=False)

        # find the edges of the board
        board.left_edge, board.right_edge = Util.find_board_edges(board.rotated_image, conf.line_profile_width,
                                                                  conf.gaussian_sigma,
                                                                  conf.approx_board_width)
        logger.info(f"{conf.source_image}: Left_edge: {board.left_edge}, Right_edge:{board.right_edge}")
        cv.imwrite(f'{conf.export.output_str}_full_rotated.png', board.rotated_image)
        cv.imwrite(f'{conf.export.output_str}_full_rotated_blue.png', board.rotated_image[:,:,2])
        logger.info(f"{conf.source_image}: Printing full rotated board")

        # find the top/bottom of the board
        board.board_width = board.right_edge - board.left_edge
        tile_width = board.board_width // 8
        Util.find_top_line(board.rotated_image, board.left_edge, board.right_edge, board)

        logger.info(
            f"{conf.source_image}: top line: {board.top_line}. Bottom line: {board.top_line + board.board_width}")
        if board.top_line < 0 or board.top_line + board.board_width > board.rotated_image.shape[0]:
            raise IndexError("The board is out of range")
        res_image = rotate(image, (board.right_tilt_angle + board.left_tilt_angle) / 2, reshape=False)
        board_edge = 0
        board.board_image = res_image[board.top_line + board_edge:board.top_line + board.board_height - board_edge,
                            board.left_edge + board_edge:board.right_edge - board_edge]
        cv.imwrite(f'{conf.export.output_str}_board.png', board.board_image)
        return board

    @staticmethod
    def set_to_min(val1: float, val2: float, max_difference: float) -> (float, float):
        if abs(val1 - val2) <= max_difference:
            return val1, val2
        if abs(val1) < abs(val2):
            return val1, val1
        return val2, val2
