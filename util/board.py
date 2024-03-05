from typing import Optional
from enum import auto

from strenum import StrEnum
import numpy as np
from pydantic import BaseModel


class Side(StrEnum):
    Black = auto()
    White = auto()


class PieceTypes(StrEnum):
    King = "K"
    Queen = "Q"
    Rock = "R"
    Bishop = "B"
    Knight = "N"
    Peon = "P"


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
    nr_of_pieces: Optional[int] = None
    fen_board: Optional[np.chararray] = np.full((8, 8), "")

    class Config:
        arbitrary_types_allowed = True

    def to_file(self, filename: str, board_name: str):
        with open(file=filename, mode='a+') as f:
            f.write(board_name + ": " + str.join(",", [piece.side[0] + piece.piece_type[0] + piece.position for piece in
                                                       self.pieces]) + "\n")

        out = ""
        for letters in range(8):
            n = 0
            for numbers in range(8):
                if self.fen_board[letters][numbers] == "":
                    n += 1
                    continue
                if n > 0:
                    out += str(n)
                    n=0
                out += str(self.fen_board[letters][numbers])
            if n > 0:
                out += str(n)
            out += "/"
        out = out[:-1]
        with open(file=filename, mode='a+') as f:
            f.write(board_name + ": " + out)