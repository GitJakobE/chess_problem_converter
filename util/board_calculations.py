from enum import StrEnum, auto
from typing import Optional
import numpy as np
from loguru import logger
from config import BoardConfig
import cv2 as cv
from . import Util
from scipy.ndimage import rotate
from dataclasses import dataclass


class Side(StrEnum):
    Black: auto()
    White: auto()


class PiecesTypes(StrEnum):
    King: auto()
    Queen: auto()
    Rock: auto()
    Bishop: auto()
    Knight: auto()
    Peon: auto()


@dataclass
class Piece:
    piece_type: PiecesTypes
    position: str
    side: Side
    image: np.ndarray


@dataclass
class Board:
    name: str = ""
    org_image: Optional[np.ndarray] = None
    board_width: Optional[int] = None
    board_height: Optional[int] = None
    rotated_image: Optional[np.ndarray] = None
    board_image: Optional[np.ndarray] = None
    left_edge: Optional[int] = None
    right_edge: Optional[int] = None
    top_line: Optional[int] = None
    right_tilt_angle: Optional[int] = None
    left_tilt_angle: Optional[int] = None
    pieces: Optional[list[Piece]] = None


class BoardCalculations:

    @staticmethod
    def find_board(image: np.ndarray, conf: BoardConfig) -> Board:
        """finds the board and returns it with the cal. images"""
        cv.imwrite(f'{conf.export.output_str}_image_org.png', image)

        board = Board(org_image=image, board_width=conf.approx_board)

        res_image = Util.remove_black_writing(image)

        # find the tilt of the board
        right_angle, left_angle = Util.find_tilt_angle(res_image, conf.line_profile_width, conf.gaussian_sigma, 10)
        board.right_tilt_angle, board.left_tilt_angle = BoardCalculations.set_to_min(right_angle, left_angle, 1.0)

        board.rotated_image = rotate(res_image, (board.right_tilt_angle + board.left_tilt_angle) / 2, reshape=False)

        # find the edges of the board
        board.left_edge, board.right_edge = Util.find_board_edges(board.rotated_image, conf.line_profile_width,
                                                                  conf.gaussian_sigma,
                                                                  conf.approx_board)
        logger.info(f"{conf.source_image}: Left_edge: {board.left_edge}, Right_edge:{board.right_edge}")
        cv.imwrite(f'{conf.export.output_str}_full_rotated.png', board.rotated_image)
        logger.info(f"{conf.source_image}: Printing full rotated board")

        # find the top/bottom of the board
        board.board_width = board.right_edge - board.left_edge
        tile_width = board.board_width // 8
        board.top_line = Util.find_top_line(board.rotated_image, board.left_edge, board.right_edge, tile_width)

        logger.info(
            f"{conf.source_image}: top line: {board.top_line}. Bottom line: {board.top_line + board.board_width}")
        if board.top_line < 0 or board.top_line + board.board_width > board.rotated_image.shape[0]:
            raise IndexError("The board is out of range")
        res_image = rotate(image, (board.right_tilt_angle + board.left_tilt_angle) / 2, reshape=False)
        board_edge = 0
        board.board_image = res_image[board.top_line + board_edge:board.top_line + board.board_width - board_edge,
                            board.left_edge + board_edge:board.right_edge - board_edge]

        return board

    @staticmethod
    def set_to_min(val1: float, val2: float, max_difference: float) -> (float, float):
        if abs(val1 - val2) <= max_difference:
            return val1, val2
        if abs(val1) < abs(val2):
            return val1, val1
        return val2, val2
