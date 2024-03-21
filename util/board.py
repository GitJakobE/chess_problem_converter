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
    Rook = "R"
    Bishop = "B"
    Knight = "N"
    Pawn = "P"


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
    l_edge: Optional[int] = None
    r_edge: Optional[int] = None
    top_line: Optional[int] = None
    r_tilt_angle: Optional[int] = None
    l_tilt_angle: Optional[int] = None
    nr_of_pieces: Optional[int] = None
    fen_board: Optional[np.chararray] = np.full((8, 8), "")

    class Config:
        arbitrary_types_allowed = True

    def to_file(self, filename: str, board_name: str):
        with open(file=filename, mode='a+') as f:
            f.write(board_name + ": " + str.join(",", [piece.side[0] + piece.piece_type[0] + piece.position for piece in
                                                       self.pieces]) + "\n")

        #printing the FEN notation
        out = ""
        for letters in range(8):
            n = 0
            for numbers in range(8):
                if self.fen_board[7-letters][numbers] == "":
                    n += 1
                    continue
                if n > 0:
                    out += str(n)
                    n=0
                out += str(self.fen_board[7-letters][numbers])
            if n > 0:
                out += str(n)
            out += "/"
        out = out[:-1]+"\n"
        with open(file=filename, mode='a+') as f:
            f.write(board_name + ": " + out)