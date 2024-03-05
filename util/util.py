import matplotlib.pylab as plt
import os
from typing import Dict, Tuple

import cv2 as cv
import numpy as np
from math import pi
from pdf2image import convert_from_path
from loguru import logger
from scipy.ndimage import gaussian_filter, rotate

from .board import Board, Piece, PieceTypes, Side
from config import BoardConfig
from . import constants


class Util:
    @staticmethod
    def find_pieces(board: Board, conf: BoardConfig) -> None:
        logger.debug(f"Printing tiles ")
        board_height, board_width, _ = board.board_image.shape
        tile_width = board_width // 8
        tile_height = board_height // 8
        tol = 3

        for y in range(8):
            for n, letter in enumerate(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']):
                tile = board.board_image[
                       board_height + tol - (y + 1) * tile_height:board_height - tol - y * tile_height,
                       tile_width * n + tol:tile_width * (1 + n) - tol]

                probabilities = conf.dm.evaluate(tile)
                if max(probabilities) < 0.4 or conf.predict_dict[np.argmax(probabilities)].split("_")[
                    -1] == 'Empty':  # empty
                    cv.imwrite(f'out/Empty/Black/{conf.export.output_str.split("/")[-1]}_{letter}{y + 1}.png',
                               tile)
                    continue
                predicted = conf.predict_dict[np.argmax(probabilities)]
                color = predicted.split("_")[0]
                piece_type = predicted.split("_")[1]
                piece = Piece(image=tile, piece_type=PieceTypes[piece_type],
                              side=Side(color), position=f"{letter}{y + 1}", probabilities=probabilities)

                cv.imwrite(f'out/{piece_type}/{color}/{conf.export.output_str.split("/")[-1]}_{letter}{y + 1}.png',
                           tile)
                cv.imwrite(f"{conf.export.output_str}_{color}_{piece_type}_{letter}{y + 1}.png", tile)
                board.pieces.append(piece)
                board.nr_of_pieces = len(board.pieces)
                board.fen_board[n][y] = PieceTypes[piece_type].value

    @staticmethod
    def img_to_parameter(img: np.array):
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        resized_gray = cv.resize(gray, (32, 32), interpolation=cv.INTER_LINEAR)
        return resized_gray.reshape(-1)

    @staticmethod
    def find_best_tile_line(image: np.ndarray, left_edge: int, right_edge: int, tile_width: int) -> list[int]:
        height, width, _ = image.shape
        min_dif = 1000
        best_i = 0
        tol = 3
        startpos = height // 2
        res = []

        while (startpos > 0):
            startpos -= tile_width
            # finding the best fit for the tiles
            for i in range(-tile_width // 2, tile_width // 2):
                line_profile = image[startpos - tile_width // 2 + i:startpos + tile_width // 2 + i,
                               left_edge:right_edge]
                tiles = []
                for x in range(8):
                    tiles.append(line_profile[:, x * tile_width + tol:(x + 1) * tile_width - tol])
                min_cross = np.min([np.std(gaussian_filter(tile, sigma=9)) for tile in tiles])

                if min_cross < min_dif:
                    best_i = i
                    min_dif = min_cross
            startpos += best_i
            res.append(startpos)
        return res

    @staticmethod
    def _find_top_line(image: np.ndarray, left_edge: int, right_edge: int, tile_width: int,
                       tile_center_line: int) -> int:
        top_reached = False
        tol = 8
        sigma = 5
        top_line = 0

        while not top_reached and tile_center_line > 0:
            tile_center_line = tile_center_line - tile_width
            line_profile = image[tile_center_line + tol - tile_width // 2:tile_center_line - tol + tile_width // 2,
                           left_edge:right_edge]
            tiles = []
            for x in range(8):
                tiles.append(line_profile[:, x * tile_width + tol + 5:(x + 1) * tile_width - tol - 5])
            if np.min([np.std(gaussian_filter(tiles[i], 5)) for i in range(8)]) * 5 >= np.std(
                    gaussian_filter(line_profile, 5)):
                top_reached = True
                top_line = tile_center_line + tile_width // 2

        return top_line

    @staticmethod
    def alt_find_top_pos(image: np.ndarray, left_edge: int, right_edge: int):
        height, width, _ = image.shape
        gray_img = image[:, :, 2]
        bgr = np.percentile(gray_img, 75)

        line_profile = gray_img[0:height // 2, left_edge:right_edge]
        stds = [np.std(line_profile[x]) for x in range(height // 2 - 1, height // 2 - 150, -1) if
                np.percentile(line_profile[x], 15) < bgr - 2]
        stds_10 = np.percentile(stds, 25)
        x = height // 2 - 150
        cons_lines = 0
        while x > 0 and (
                cons_lines < 5 or (np.percentile(line_profile[x], 15) < bgr - 2) or np.std(line_profile[x]) > stds_10):
            x -= 1
            cons_lines = cons_lines + 1 if np.percentile(line_profile[x], 15) >= bgr - 2 and np.std(
                line_profile[x]) <= stds_10 else 0
        return x + 8

    @staticmethod
    def locate_board_from_template(image: np.ndarray, board: Board) -> (int, int):
        bgr = np.percentile(image, 75)
        _, width = image.shape
        image[image[:, :] < bgr - 10] = bgr - 10

        best_res = -1000
        best_width = 0
        best_height = 0
        for tile_width in range(60, 69):
            for tile_height in range(60, 69):
                template = np.full((tile_height * 8, tile_width * 8), bgr)
                template = Util.create_board(template, tile_width, tile_height, bgr)
                res = cv.matchTemplate(image, template.astype(np.uint8), cv.TM_CCOEFF_NORMED)
                if (res.max() > best_res):
                    best_res = res.max()
                    best_width, best_height = tile_width, tile_height
                    topline, left_side = np.where(res == res.max())
        board.board_width = best_width * 8
        board.board_height = best_height * 8
        return topline[0], left_side[0]

    @staticmethod
    def find_top_line(image: np.ndarray, board: Board) -> (int, int):
        height, width, _ = image.shape
        side_cut = 30
        top_cut = 50
        bottom_cut = 100
        top, leftside = Util.locate_board_from_template(
            image=image[top_cut:height - bottom_cut, side_cut:width - side_cut, 2], board=board)
        board.left_edge = side_cut + leftside
        board.top_line = top + top_cut
        board.right_edge = board.left_edge + board.board_width
        return top, leftside

    @staticmethod
    def remove_black_writing(image):
        height, width, _ = image.shape
        res_image = np.copy(image)
        bgr = np.median(image)
        th1 = bgr * 0.90
        th2 = 150
        th3 = 100
        # remove black:
        for x in range(1, width - 1):
            for y in range(1, height - 1):
                if (np.all(res_image[y][x] < th1) and np.min(res_image[y][x]) * 1.1 > np.max(res_image[y][x])) or (
                        np.all(res_image[y][x] < th2) and np.min(res_image[y][x]) * 1.2 > np.max(
                    res_image[y][x])) or np.all(res_image[y][x] < th3):
                    res_image[y][x] = np.percentile(res_image[y - 1:y + 2, x - 1:x + 2, :], 75, axis=(0, 1))

        return res_image

    @staticmethod
    def find_board_edges(image, line_profile_width, gaussian_sigma, approx_board):
        height, width, _ = image.shape
        side_cut = 5

        gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        flattened_profile = np.mean(
            gray_img[(height - line_profile_width) // 2:(height + line_profile_width) // 2,
            side_cut:width - side_cut], axis=0)

        smoothed_profile = gaussian_filter(flattened_profile, sigma=3)

        max_div = 0
        max_tol = 40
        pos_left = 0
        pos_right = width - 1

        # Calculate the derivative
        for x in range(100):
            # matching the left and right side derivatives to find the lowest point
            for tol in range(-max_tol, max_tol):
                if (x + approx_board + 1 + tol) < len(smoothed_profile):
                    derivative = smoothed_profile[x] - smoothed_profile[x + 1] - smoothed_profile[
                        x + approx_board + tol] + smoothed_profile[x + approx_board + 1 + tol]
                    if derivative > max_div:
                        max_div = derivative
                        pos_left = x
                        pos_right = x + approx_board + tol

        pos_left = pos_left + side_cut + 5
        pos_right = pos_right + side_cut - 5

        return pos_left, pos_right

    @staticmethod
    def find_tilt_angle(image, line_profile_width=300, gaussian_sigma=5, tilt_range=10):
        height, width, _ = image.shape
        h_line_profile_width = line_profile_width // 2
        edge_tol = 10
        # Select the area for the line profile based on the edge we are processing
        profile_slice = image[height // 2 - h_line_profile_width:height // 2 + h_line_profile_width,
                        edge_tol:width - edge_tol]

        # Calculate the average color across the vertical axis to get the profile
        profile = np.mean(profile_slice, axis=2)

        # Initialize variables to find the maximum derivative and its corresponding angle
        right_max_derivative_value = left_max_derivative_value = 0
        right_max_derivative_angle = left_max_derivative_angle = 0
        # Rotate the profile within the specified range and find the angle with the highest first derivative
        for angle in np.arange(-tilt_range, tilt_range + 1, 0.25):
            # Rotate the profile
            rotated_profile = rotate(profile, angle, reshape=False)

            flattened_profile = np.median(rotated_profile, axis=0)
            # Apply Gaussian filter
            smoothed_profile = gaussian_filter(flattened_profile, sigma=gaussian_sigma)

            # Calculate the derivative
            derivative = np.gradient(smoothed_profile, axis=0)
            left_max_derivative = np.min(derivative[:150])
            right_max_derivative = np.max(derivative[150:])

            # If the new derivative is greater than the max, update the max derivative and angle
            if left_max_derivative > left_max_derivative_value:
                left_max_derivative_value = left_max_derivative
                left_max_derivative_angle = angle
            if right_max_derivative > right_max_derivative_value:
                right_max_derivative_value = right_max_derivative
                right_max_derivative_angle = angle

        return right_max_derivative_angle, left_max_derivative_angle

    @staticmethod
    def locate_templates(image_gray):
        """ Locate points where the templates appear in image_gray"""
        templates = [r'data\wpawn.png', r'data\whorse.png', r'data\bpawn.png', r'data\bhorse.png']
        locations = []
        for template_name in templates:
            template = cv.imread(template_name)
            template_gray = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
            w, h = template_gray.shape[::-1]
            res = cv.matchTemplate(image_gray, template_gray, cv.TM_CCOEFF_NORMED)
            threshold = 0.8
            locations.append({'name': template_name, 'w': w, 'h': h, 'positions': np.where(res >= threshold)})

        return locations

    @staticmethod
    def split_black_white_pieces(folder_path: str):
        files = [file.name for file in os.scandir(folder_path) if file.is_file()]
        for file in files:
            image = cv.imread(folder_path + "\\" + file)
            # if more than 10% are dominately red we have a white piece:
            folder = "Black"
            if (np.sum(image[:, :, 0] * 1.5 < image[:, :, 2]) > (image.shape[0] * image.shape[1] * 0.05)):
                folder = "White"
                print(f"{file} is white")
            if os.path.exists(folder_path + f"\\{folder}\\" + file):
                os.remove(folder_path + f"\\" + file)
            else:
                os.rename(folder_path + "\\" + file, folder_path + f"\\{folder}\\" + file)
            print(f"{file} is black")

    @staticmethod
    def convert_pdf_to_pngs(in_path: str, out_path: str = "") -> None:
        if out_path == "":
            out_path = in_path + "/png"
            if not os.path.exists(out_path):
                os.makedirs(out_path)
        filenames = Util.get_all_pdfs_filenames_in_paths(in_path)

        for filename in filenames:
            try:
                pages = convert_from_path(filename, 600)
                for n, page in enumerate(pages):
                    out_filename = filename.replace(in_path, out_path)

                    if n == 0:
                        out_filename = out_filename.replace(".pdf", f".png")
                        page.resize(constants.standard_image_dim)
                        page.save(out_filename)
                    logger.info(f"converted: {filename} to {out_filename}")
            except Exception as e:
                logger.error(f"Error in {filename}: {e}")

    @staticmethod
    def get_all_pdfs_filenames_in_paths(path: str) -> list[str]:

        return [filename.path for filename in os.scandir(path) if filename.path.endswith(".pdf")]

    @staticmethod
    def draw_locations(target_image, locations):
        """Draw template locations"""
        for template in locations:
            for pt in zip(*template['positions'][::-1]):
                cv.rectangle(target_image, pt, (pt[0] + template['w'], pt[1] + template['h']), (0, 255, 255), 2)

    @staticmethod
    def find_horizontal_vertical_lines(image_gray):
        """Locate all lines < 15 degrees rotated"""
        edges = cv.Canny(image_gray, 50, 150, apertureSize=3)
        lines = cv.HoughLines(edges, 1, np.pi / 180, 200)
        # singleDegInRadians = pi / 180.0
        max_rotation_degrees = 15.0
        max_rotation_radians = max_rotation_degrees / 180.0 * pi
        horizontal_lines = []
        vertical_lines = []

        for i in range(len(lines)):
            for rho, theta in lines[i]:
                if abs(theta) < max_rotation_radians or abs(theta - pi) < max_rotation_radians:
                    vertical_lines.append(lines[i])
                if abs(theta - pi / 2.0) < max_rotation_radians or abs(theta - 3.0 * pi / 2.0) < max_rotation_radians:
                    horizontal_lines.append(lines[i])
        return horizontal_lines, vertical_lines

    @staticmethod
    def points_for_line(rho, theta):
        """Returns points for hough line defined by rho and theta"""
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 10000 * (-b))
        y1 = int(y0 + 10000 * a)
        x2 = int(x0 - 10000 * (-b))
        y2 = int(y0 - 10000 * a)
        return x1, y1, x2, y2

    @staticmethod
    def draw_lines(target_image, lines):
        """Draw hough lines on targetImage"""
        for i in range(len(lines)):
            for rho, theta in lines[i]:
                x1, y1, x2, y2 = Util.points_for_line(rho, theta)

                cv.line(target_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    @staticmethod
    def unify_angle(a):
        """Move angle into [-PI,PI]"""
        if a > pi / 2.0:
            return a - pi
        return a

    @staticmethod
    def create_board(image: np.ndarray, tile_width: int, tile_height: int, bgr: int) -> np.ndarray:
        height, width = image.shape
        if width // 2 - tile_width * 4 < 0:
            print("out of range")
        black = bgr - 10
        start_x = width // 2 - tile_width * 4
        start_y = height // 2 - tile_height * 4
        for tile_x in range(8):
            for tile_y in range(8):
                base_tile_color = bgr
                if (tile_x % 2 == 0 and tile_y % 2 != 0) or (tile_x % 2 != 0 and tile_y % 2 == 0):
                    base_tile_color = black

                for x in range(tile_width):
                    for y in range(tile_height):
                        tile_color = bgr
                        if base_tile_color == black and ((x % 2 == 0 and y % 2 != 0) or (x % 2 != 0 and y % 2 == 0)):
                            tile_color = black
                        image[start_y + y + tile_y * tile_height][start_x + x + tile_x * tile_width] = base_tile_color
        return image

    @staticmethod
    def match_folders_to_piece_types(path: str) -> Dict[Tuple[Side, PieceTypes], str]:
        res = {}
        for dir in os.scandir(path=path):
            if dir.is_dir() and dir.name in PieceTypes.__members__:
                for subdir in os.scandir(path=dir):
                    if subdir.is_dir() and subdir.name in Side.__members__:
                        res[(Side[subdir.name], PieceTypes[dir.name])] = subdir.path
        return res
