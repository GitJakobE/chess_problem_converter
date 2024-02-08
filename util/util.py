import cv2 as cv
import numpy as np
from math import pi
from pdf2image import convert_from_path
from loguru import logger
from scipy.ndimage import gaussian_filter, rotate
from pathlib import Path
from sklearn import pipeline

from chess_problem_converter.config import BoardConfig


class Util:
    @staticmethod
    def print_tiles(board: np.ndarray, conf: BoardConfig, classifier: pipeline = None) -> None:
        logger.debug(f"Printing tiles ")
        _, board_width, _ = board.shape
        tile_width = board_width // 8
        tol = 3
        std_threshold = 15


        for y in range(8):
            for n, letter in enumerate(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']):
                tile = board[board_width + tol - (y + 1) * tile_width:board_width - tol - y * tile_width,
                       tile_width * n + tol:tile_width * (1 + n) - tol]
                if (conf.only_pieces):
                    if std_threshold > np.std(gaussian_filter(tile, 5)):
                        continue
                if classifier is not None:
                    predicted = classifier.predict(Util.img_to_parameter(tile).reshape(1,-1))[0]
                    predicted = conf.predict_dict[predicted]
                    color = predicted.split("_")[0]
                    piece = predicted.split("_")[1]
                    cv.imwrite(f'out/{piece}/{color}/{conf.export.output_str.split("/")[-1]}_{letter}{y + 1}.png', tile)
                else:
                    cv.imwrite(f'{conf.export.output_str}_{letter}{y + 1}.png', tile)

    @staticmethod
    def img_to_parameter(img: np.array):
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        resized_gray = cv.resize(gray, (32, 32), interpolation=cv.INTER_LINEAR)
        return resized_gray.reshape(-1)

    @staticmethod
    def find_top_line(image, left_edge, right_edge, tile_width):
        height, width, _ = image.shape

        # find the middel of a square/tile
        min_dif = 1000
        best_i = 0
        tol = 3

        for i in range(tile_width):
            line_profile = image[(height - tile_width) // 2 + i:(height + tile_width) // 2 + i,
                           left_edge:right_edge]
            tiles = []
            for x in range(8):
                tiles.append(line_profile[:, x * tile_width + tol:(x + 1) * tile_width - tol])
            min_cross = np.min([np.std(gaussian_filter(tile, sigma=9)) for tile in tiles])

            if min_cross < min_dif:
                best_i = i
                min_dif = min_cross
        y_line = height // 2 + best_i
        cv.imwrite(f'out/line_profile.png', image[y_line - tile_width // 2 + 5:y_line - 5 + tile_width // 2,
                                            left_edge:right_edge])
        top_reached = False

        top_line = 0
        while not top_reached:
            y_line = y_line - tile_width
            line_profile = image[y_line + 5:y_line - 5 + tile_width // 2,
                           left_edge:right_edge]
            tiles = []
            for x in range(8):
                tiles.append(line_profile[:, x * tile_width + tol + 5:(x + 1) * tile_width - tol - 5])
            if np.min([np.std(gaussian_filter(tiles[i], 5)) for i in range(8)]) * 5 > np.std(
                    gaussian_filter(line_profile, 5)):
                top_reached = True
                top_line = y_line + tile_width // 2

        return top_line

    @staticmethod
    def remove_black_writing(image):
        height, width, _ = image.shape
        th = 150
        # remove black:
        for x in range(width):
            for y in range(height):
                if np.all(image[y][x] < th):
                    for color in range(3):
                        image[y][x][color] = 230
        return image

    @staticmethod
    def find_board_edges(image, line_profile_width, gaussian_sigma, approx_board):
        height, width, _ = image.shape
        side_cut = 5

        flattened_profile = np.mean(
            image[(height - line_profile_width) // 2:(height + line_profile_width) // 2,
            side_cut:width - side_cut], axis=0)

        smoothed_profile = gaussian_filter(flattened_profile, sigma=gaussian_sigma)

        max_div = 0
        max_tol = 20
        pos_left = 0
        pos_right = width - 1

        # Calculate the derivative
        for x in range(max_tol, len(smoothed_profile) - max_tol - approx_board):
            for tol in range(-max_tol, max_tol):
                derivative = np.mean(
                    smoothed_profile[x] - smoothed_profile[x + 1] + smoothed_profile[x + approx_board + tol] -
                    smoothed_profile[
                        x + approx_board - 1 + tol])
                if derivative > max_div:
                    max_div = derivative
                    pos_left = x
                    pos_right = x + approx_board + tol
        pos_left = pos_left + int(gaussian_sigma / 2) + side_cut
        pos_right = pos_right + int(gaussian_sigma / 2) + side_cut
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
        for angle in np.arange(-tilt_range, tilt_range + 1):
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
            cv.imwrite(f'../out/test{angle}.png', rotated_profile)
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
    def convert_pdf_to_pngs(path: str) -> None:

        pages = convert_from_path(path, 500)
        n = 0
        filname = str.replace(path, ".pdf", "")
        for page in pages:
            page.save(f"{filname}--{n}.png")
            n += 1

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
    def rotate_image(image, angle):
        """Rotate an image by angle degrees"""
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)
        return result

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
    def find_average_rotation(horizontal_lines, vertical_lines):
        """Determines average rotation from a set of horizontal and vertical lines"""
        total_lines = len(horizontal_lines) + len(vertical_lines)
        if total_lines < 1:
            return 0
        avg_theta = 0.0
        for h_l in horizontal_lines:
            avg_theta += Util.unify_angle(h_l[0][1] - pi / 2.0)
        for v_l in vertical_lines:
            avg_theta += Util.unify_angle(v_l[0][1])

        return avg_theta / total_lines

    @staticmethod
    def find_bounding_box(horizontal_lines, vertical_lines):
        """return BB of lines (min_x, min_y, max_x, max_y)"""
        min_x = 1024
        max_x = 0
        min_y = 1024
        max_y = 0
        for h_l in horizontal_lines:
            _, y, _, _ = Util.points_for_line(h_l[0][0], h_l[0][1])
            min_y = min(min_y, y)
            max_y = max(max_y, y)
        for v_l in vertical_lines:
            x, _, _, _ = Util.points_for_line(v_l[0][0], v_l[0][1])
            min_x = min(min_x, x)
            max_x = max(max_x, x)
        return min_x, min_y, max_x, max_y

    @staticmethod
    def setup_grid():
        """Return unfilled grid [8][8] of empty string"""
        grid = []
        for y in range(8):
            grid_row = []
            for x in range(8):
                grid_row.append("")
            grid.append(grid_row)
        return grid

    @staticmethod
    def fill_templates_in_grid(grid, bounding_box, template_locations):
        """Goes through each cell and set the content of the cell to the (last) template the was found in the cell"""
        x1 = bounding_box[0]
        w = bounding_box[2] - bounding_box[0]
        y1 = bounding_box[1]
        h = bounding_box[3] - bounding_box[1]
        for y in range(8):
            for x in range(8):
                gx1 = x1 + w / 8.0 * x
                gy1 = y1 + h / 8.0 * y
                gx2 = gx1 + w / 8.0
                gy2 = gy1 + h / 8.0
                for template in template_locations:
                    for pt in zip(*template['positions'][::-1]):
                        if gx1 < pt[0] < gx2 and gy1 < pt[1] < gy2:
                            grid[y][x] = template['name']
        return grid
