import cv2 as cv
import numpy as np
from math import pi
from pdf2image import convert_from_path


class Util:
    @staticmethod
    def locate_templates(image_gray):
        """ Locate points where the templates appear in image_gray"""
        templates = ['wpawn.png', 'whorse.png', 'bpawn.png', 'bhorse.png']
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
    def convert_pdf_to_pngs(filename: str) -> None:
        pages = convert_from_path(filename, 500)
        n = 0
        filename = str.replace(filename, ".pdf", "")
        for page in pages:
            page.save(f"{filename}--{n}.png")
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
        return x0, y0, x1, y1, x2, y2

    @staticmethod
    def draw_lines(target_image, lines):
        """Draw hough lines on targetImage"""
        for i in range(len(lines)):
            for rho, theta in lines[i]:
                x0, y0, x1, y1, x2, y2 = Util.points_for_line(rho, theta)

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
            _, _, _, y, _, _ = Util.points_for_line(h_l[0][0], h_l[0][1])
            min_y = min(min_y, y)
            max_y = max(max_y, y)
        for v_l in vertical_lines:
            _, _, x, _, _, _ = Util.points_for_line(v_l[0][0], v_l[0][1])
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
