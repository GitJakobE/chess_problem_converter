from typing import Optional
from enum import auto

from strenum import StrEnum
import numpy as np
from pydantic import BaseModel


class Side(StrEnum):
    Black = auto()
    White = auto()


class PieceTypes(StrEnum):
    King = auto()
    Queen = auto()
    Rock = auto()
    Bishop = auto()
    Knight = auto()
    Peon = auto()


class Piece(BaseModel):
    piece_type: PieceTypes
    position: str
    side: Side
    image: np.ndarray
    probabilities: list[float]

    class Config:
        arbitrary_types_allowed = True


class Board(BaseModel):
    name: str = ""
    pieces: list[Piece] = []
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
    nr_of_pieces: Optional[int] =None

    class Config:
        arbitrary_types_allowed = True

    def to_file(self, filename: str, board_name: str):
        with open(file=filename, mode='a+') as f:
            f.write(board_name + ": " + str.join(",", [piece.side[0] + piece.piece_type[0] + piece.position for piece in
                                                       self.pieces]) + "\n")
