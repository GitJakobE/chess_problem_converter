import os

import numpy as np

from util.board import Board, Side, Piece, PieceTypes


def test_side() -> None:
    w_test = Side("White")
    b_test = Side("Black")
    assert w_test.name == "White"
    assert b_test.name == "Black"


def test_piece_types() -> None:
    king_test = PieceTypes("King")
    queen_test = PieceTypes("Queen")
    bishop_test = PieceTypes("Bishop")
    rook_test = PieceTypes("Rook")
    knight_test = PieceTypes("Knight")
    pawn_test = PieceTypes("Pawn")
    assert queen_test.name == "King"
    assert king_test.name == "Queen"
    assert bishop_test.name == "Bishop"
    assert rook_test.name == "Rook"
    assert knight_test.name == "Knight"
    assert pawn_test.name == "Pawn"


def test_pieces() -> None:
    image = np.zeros((64, 64))
    piece = Piece(piece_type=PieceTypes('King'), side=Side('White'), position="A1", image=image)
    assert piece.side == Side.White
    assert piece.position == "A1"
    assert piece.piece_type == PieceTypes.King
    assert np.array_equal(piece.image, image)


def test_board_model():
    image = np.zeros((64, 64))
    piece = Piece(piece_type=PieceTypes.King, position="A1", side=Side.White, image=image)
    board = Board(name="Test Board", pieces=[piece], org_image=image)
    assert board.name == "Test Board"
    assert len(board.pieces) == 1
    assert np.array_equal(board.org_image, image)


def test_board_to_file():
    image = np.zeros((100, 100))
    piece = Piece(piece_type=PieceTypes.King, position="E4", side=Side.White, image=image)
    board = Board(name="Test Board", pieces=[piece, piece])
    file_path = "../../temp/board.txt"
    board.to_file(file_path, "Test Board")
    with open(file_path) as f:
        content = f.read()
    os.unlink(file_path)
    expected_content = "Test Board: WKE4,WKE4\n"
    assert content == expected_content
