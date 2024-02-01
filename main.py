import cv2 as cv
import sys
from util import Util
from scipy.ndimage import rotate
from loguru import logger
from config import BoardConfig

source_image = "data\Toft-00-0051.png" if len(sys.argv) < 2 else sys.argv[1]


# Main function
def main():
    conf = BoardConfig()
    conf.set_export(source_image)

    image = cv.imread(source_image)

    cv.imwrite(f'{conf.export.output_str}_image_org.png', image)

    # find the tilt of the board
    right_tilt_angle, left_tilt_angle = Util.find_edge(image, conf.line_profile_width, conf.gaussian_sigma, 10)
    rotated_image = rotate(image, (right_tilt_angle + left_tilt_angle) / 2, reshape=False)

    # find the edges of the board
    left_edge, right_edge = Util.find_board_edges(rotated_image, conf.line_profile_width, conf.gaussian_sigma,
                                                  conf.approx_board)
    image = Util.remove_black_writing(rotated_image)
    cv.imwrite(f'{conf.export.output_str}_full_rotated.png', image)

    # find the top/bottom of the board
    board_width = right_edge - left_edge
    tile_width = board_width // 8
    top_line = Util.find_top_line(image, left_edge, right_edge, tile_width)

    board = image[top_line + 10:top_line + board_width - 10, left_edge + 10:right_edge - 10]

    Util.print_board(board, conf, True)


# Run the program
main()
