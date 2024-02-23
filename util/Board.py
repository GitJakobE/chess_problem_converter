from enum import auto
from strenum import StrEnum
from typing import Optional
import numpy as np
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
